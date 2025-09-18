import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

from vector_quantize_pytorch import VectorQuantize

# -------------------
# Setup and Data Loading
# -------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor()
])
train_set = torchvision.datasets.STL10(root="./data", split="train", download=True, transform=transform)
test_set  = torchvision.datasets.STL10(root="./data", split="test",  download=True, transform=transform)

batch_size = 64
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


class MyVectorQuantize(nn.Module):
    """
    A simplified custom Vector Quantizer (VQ) layer with EMA updates for stable training.
    
    This class quantizes input vectors by mapping them to the nearest codebook entries,
    using Exponential Moving Average (EMA) to update the codebook during training.
    It also includes optional dead code revival to prevent unused codes.
    
    Args:
        dim (int): Dimensionality of each vector (e.g., feature depth).
        codebook_size (int): Number of vectors in the codebook (vocabulary size).
        decay (float): EMA decay factor (higher = slower updates, default 0.99).
        commitment_weight (float): Weight for commitment loss (default 0.25).
        epsilon (float): Small value to avoid division by zero (default 1e-5).
        threshold_ema_dead_code (int): Threshold for considering a code "dead" (default 2).
    """
    def __init__(self, dim, codebook_size, decay=0.99, commitment_weight=0.25, epsilon=1e-5, threshold_ema_dead_code=2):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.decay = decay
        self.commitment_weight = commitment_weight
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code
        
        # Codebook: Learnable embedding layer storing quantized vectors
        self.embedding = nn.Embedding(self.codebook_size, self.dim)
        
        # EMA buffers: Track cluster sizes and smoothed embeddings for stable updates
        self.register_buffer('ema_cluster_size', torch.zeros(codebook_size))
        self.register_buffer('ema_embeddings', self.embedding.weight.data.clone())
        self.register_buffer('initialized', torch.tensor(False, dtype=torch.bool))
    
    @torch.no_grad()
    def _reset_dead_codes(self, x):
        """
        Optional: Revive "dead" codes (those used below the threshold) by replacing them
        with random samples from the current batch. This prevents codebook collapse.
        """
        dead_mask = self.ema_cluster_size < self.threshold_ema_dead_code
        num_dead = dead_mask.sum().item()
        
        if num_dead > 0:
            # Sample replacement vectors from the batch
            batch_samples = x[torch.randperm(x.size(0))[:num_dead]]
            replacements = batch_samples.to(self.embedding.weight.dtype)
            
            # Update codebook and EMA buffers
            self.embedding.weight.data[dead_mask] = replacements
            self.ema_embeddings.data[dead_mask] = replacements
            self.ema_cluster_size[dead_mask] = 1.0
    
    def forward(self, z_e_flat):
        """
        Forward pass: Quantize input vectors and compute losses.
        
        Args:
            z_e_flat (Tensor): Encoder output, shape (B, N, D) where B=batch, N=tokens, D=dim.
        
        Returns:
            Tuple[Tensor, Tensor, Tensor]: (quantized vectors, indices, commitment loss).
        """
        # Step 1: Prepare inputs
        z_e_flat = z_e_flat.float()  # Ensure float32 for numerical stability
        B, N, D = z_e_flat.shape
        z_flat = z_e_flat.reshape(-1, D)  # Flatten to (B*N, D) for vectorized ops

        
        # Step 2: Compute distances to codebook vectors
        # Use squared Euclidean distance: ||z - e||^2 = z^2 + e^2 - 2*z*e
        z_sq = torch.sum(z_flat.pow(2), dim=1, keepdim=True)  # (B*N, 1)
        e_sq = torch.sum(self.embedding.weight.pow(2), dim=1)  # (codebook_size,)
        z_dot_e = torch.matmul(z_flat, self.embedding.weight.t())  # (B*N, codebook_size)
        distances = z_sq - 2 * z_dot_e + e_sq  # (B*N, codebook_size)
        
        # Step 3: Find nearest codebook vectors
        indices_flat = torch.argmin(distances, dim=-1)  # (B*N,)
        z_q_flat = self.embedding(indices_flat)  # (B*N, D)
        
        # Step 4: Update codebook via EMA (only during training)
        if self.training:
            # One-hot encode indices for aggregation
            encodings = F.one_hot(indices_flat, self.codebook_size).float()  # (B*N, codebook_size)
            cluster_counts = encodings.sum(0)  # (codebook_size,)
            dw = torch.matmul(encodings.t(), z_flat)  # (codebook_size, D)
            
            # Update EMA buffers (restore the missing update for ema_cluster_size)
            self.ema_cluster_size.data.mul_(self.decay).add_(cluster_counts, alpha=1 - self.decay)
            self.ema_embeddings.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)
            
            # Normalize and copy to embedding (now use EMA cluster size for smoothing)
            normalized_embeds = self.ema_embeddings / (self.ema_cluster_size.unsqueeze(1) + self.epsilon)  
                    
            self._reset_dead_codes(z_flat)

        # Step 5: Compute commitment loss (encourages encoder to match quantized vectors)
        commit_loss = self.commitment_weight * F.mse_loss(z_e_flat, z_q_flat.view(B, N, D).detach())
        
        # Step 6: Apply Straight-Through Estimator (STE) for gradient flow
        z_q = z_e_flat + (z_q_flat.view(B, N, D) - z_e_flat).detach()
        indices = indices_flat.view(B, N)
        
        return z_q, indices, commit_loss

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1),
        )
    def forward(self, x): return F.relu(x + self.net(x), inplace=True)

class Encoder(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, 1), nn.ReLU(inplace=True),   # 96 -> 48
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(inplace=True), # 48 -> 24
            nn.Conv2d(256, dim, 3, 1, 1), nn.ReLU(inplace=True),
            ResBlock(dim),
            ResBlock(dim)
        )
    
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.net = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            nn.ConvTranspose2d(dim, 256, 4, 2, 1), nn.ReLU(inplace=True), # 24 -> 48
            ResBlock(256),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(inplace=True), # 48 -> 96
            ResBlock(128),
            nn.Conv2d(128, 3, 3, 1, 1), nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

class VAE(nn.Module):
    """
    VAE using shared Encoder/Decoder, with middle: flatten -> linear stats -> sample -> linear -> unflatten.
    """
    def __init__(self, dim=128, latent_dim=512):
        super().__init__()
        self.enc = Encoder(dim)
        self.dec = Decoder(dim)
        self.dim = dim
        self.latent_dim = latent_dim
        self.to_stats = nn.Linear(dim * 24 * 24, latent_dim * 2)
        self.from_latent = nn.Linear(latent_dim, dim * 24 * 24)

    def forward(self, x, beta=1.0):
        h = self.enc(x)  # (B, dim, 24, 24)
        stats = self.to_stats(h.flatten(1))
        mu, logvar = torch.chunk(stats, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        h_dec = self.from_latent(z).view(-1, self.dim, 24, 24)
        x_hat = self.dec(h_dec)
        recon = F.l1_loss(x_hat, x, reduction='mean')
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return x_hat, recon + beta * kld

class VQVAE(nn.Module):
    """
    VQ-VAE using shared Encoder/Decoder, with middle: permute/reshape -> VQ -> reshape/permute.
    """
    def __init__(self, dim=128, codebook_size=1024, custom=False):
        super().__init__()
        self.enc = Encoder(dim)
        self.dec = Decoder(dim)
        if custom:
            self.vq = MyVectorQuantize(dim=dim, codebook_size=codebook_size, decay=0.99, commitment_weight=0.25)
        else:
            self.vq = VectorQuantize(dim=dim, codebook_size=codebook_size, decay=0.99, commitment_weight=0.25)

    def forward(self, x):
        z_e = self.enc(x)  # (B, dim, 24, 24)
        B, C, H, W = z_e.shape
        z_e_flat = z_e.permute(0, 2, 3, 1).reshape(B, H*W, C)
        z_q, indices, commit_loss = self.vq(z_e_flat)
        z_q = z_q.view(B, H, W, C).permute(0, 3, 1, 2)
        x_hat = self.dec(z_q)
        recon = F.l1_loss(x_hat, x, reduction='mean')
        return x_hat, recon + commit_loss, (H*W, self.vq.codebook_size, indices)

class FSQ(nn.Module):
    """
    FSQ using shared Encoder/Decoder, with middle: quantize.
    """
    def __init__(self, dim=128, levels=16):
        super().__init__()
        self.enc = Encoder(dim)
        self.dec = Decoder(dim)
        self.levels = levels

    @staticmethod
    def ste_round(x):
        return (x.round() - x).detach() + x

    def quantize(self, z):
        z = torch.tanh(z)
        s = self.levels / 2.0
        z_q = self.ste_round(s * z) / s
        return z_q.clamp(-1, 1)

    def forward(self, x):
        z = self.enc(x)  # (B, dim, 24, 24)
        z_q = self.quantize(z)
        x_hat = self.dec(z_q)
        recon = F.l1_loss(x_hat, x, reduction='mean')
        commit = 0.25 * F.mse_loss(torch.tanh(z), z_q.detach())
        return x_hat, recon + commit, (z_q.shape[-2]*z_q.shape[-1], self.levels, None)

# -------------------
# Training Utilities
# -------------------
epochs = 12
lr = 3e-4
scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

def train_epoch(model, loader, opt, epoch, name, total_epochs, pbar):
    model.train()
    total = 0.0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            if isinstance(model, VAE):
                # KL annealing: start small, increase to 1.0
                beta = min(1.0, epoch / max(1, total_epochs//3))
                _, loss = model(x, beta=beta)
            else:
                _, loss, _ = model(x)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        total += loss.item() * x.size(0)
        pbar.update(1)
    return total / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, epoch, total_epochs, pbar):
    model.eval()
    total = 0.0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            if isinstance(model, VAE):
                beta = 1.0
                _, loss = model(x, beta=beta)
            else:
                _, loss, _ = model(x)
        total += loss.item() * x.size(0)
        pbar.update(1)
    return total / len(loader.dataset)

def train_and_eval(model, name):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, test_losses = [], []
    total_train_steps = epochs * len(train_loader)
    total_eval_steps = epochs * len(test_loader)
    print(f"Starting training for {name}")
    train_pbar = tqdm(total=total_train_steps, desc=f"{name} Train", position=0)
    eval_pbar = tqdm(total=total_eval_steps, desc="Eval", position=1)
    for ep in range(1, epochs+1):
        tr = train_epoch(model, train_loader, opt, ep, name, epochs, train_pbar)
        te = eval_epoch(model, test_loader, ep, epochs, eval_pbar)
        train_losses.append(tr); test_losses.append(te)
        train_pbar.set_postfix(epoch=f"{ep:02d}/{epochs}", train_loss=f"{tr:.4f}", test_loss=f"{te:.4f}")
    train_pbar.close()
    eval_pbar.close()
    return train_losses, test_losses, model

# -------------------
# Run Training
# -------------------
results = {}
results['VAE'] = train_and_eval(VAE(latent_dim=512), "VAE")
results['VQ-VAE'] = train_and_eval(VQVAE(dim=128, codebook_size=512, custom=False), "VQ-VAE")
results['MyVQ-VAE'] = train_and_eval(VQVAE(dim=128, codebook_size=512, custom=True), "MyVQ-VAE")
results['FSQ'] = train_and_eval(FSQ(dim=128, levels=16), "FSQ")

# -------------------
# Plot Loss Curves
# -------------------
plt.figure(figsize=(10, 6))
for name, (tr, te, _) in results.items():
    plt.plot(tr, label=f"{name} train")
    plt.plot(te, '--', label=f"{name} test")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train & Test Loss (STL10, 96x96)")
plt.legend()
plt.grid(True)
plt.savefig('loss_curves.png')

# -------------------
# Visualization
# -------------------
@torch.no_grad()
def show_one(results):
    x, _ = next(iter(test_loader))
    x = x[:1].to(device)
    orig = x[0].permute(1, 2, 0).cpu().numpy()

    for name, (_, _, model) in results.items():
        model.eval()
        fig, axs = plt.subplots(1, 3, figsize=(9, 3))
        
        if isinstance(model, VQVAE):
            x_hat, _, info = model(x)
            tokens, vocab, indices = info
            title = f"{name}\nTokens={tokens}, Vocab={vocab}"
        elif isinstance(model, FSQ):
            x_hat, _, info = model(x)
            tokens, levels, _ = info
            title = f"{name}\nTokens={tokens}, Levels={levels}"
        else:  # VAE
            x_hat, _ = model(x, beta=1.0)
            title = f"{name}\nLatent={model.latent_dim}, Vocab=N/A"

        dec = x_hat[0].permute(1, 2, 0).cpu().numpy()
        
        axs[0].imshow(orig)
        axs[0].set_title("Original")
        axs[0].axis("off")

        axs[1].text(0.5, 0.5, title, ha="center", va="center", fontsize=11)
        axs[1].axis("off")

        axs[2].imshow(dec)
        axs[2].set_title("Decoded")
        axs[2].axis("off")

        plt.tight_layout()
        filename = name.lower().replace(' ', '').replace('-', '') + '.png'
        plt.savefig(filename)
        plt.close(fig)  # Close to free memory

show_one(results)