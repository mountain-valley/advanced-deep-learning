# main.py
import os
import argparse
from datetime import datetime
import json
import warnings
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import torchvision.transforms.v2 as v2
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from transformers import AutoModel, AutoConfig

warnings.filterwarnings("ignore", category=UserWarning)

# --- Data Handling ---
class SimCLRImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, _ = self.samples[index]
        sample = self.loader(path)
        view1 = self.transform(sample)
        view2 = self.transform(sample)
        return view1, view2


def get_simclr_transform(size):
    return v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(size=size, antialias=True),
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        # v2.RandomHorizontalFlip(p=0.5),
        # v2.RandomRotation(degrees=90),
        # v2.RandomGrayscale(p=0.2),
        # v2.ToDtype(torch.float32, scale=True),
        # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

def get_supervised_transform(size, is_train=True):
    if is_train:
        return T.Compose([
            T.Resize(size), T.RandomResizedCrop(size=size), T.RandomHorizontalFlip(), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize((size, size)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

def prepare_imagenet_loader(args, simclr=False):
    # Now we always treat --imagenet_path as tiny-imagenet-200 by default
    base = args.imagenet_path
    train_dir = os.path.join(base, 'train')
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"tiny-imagenet-200 'train' directory not found at: {train_dir}")
    
    dataset = SimCLRImageFolder(root=train_dir, transform=get_simclr_transform(args.image_size)) if simclr else \
              ImageFolder(root=train_dir, transform=get_supervised_transform(args.image_size, is_train=True))

    # For SimCLR, prefer dropping the last undersized batch to keep pairs consistent
    drop_last = True if simclr else False
    return DataLoader(dataset, batch_size=args.pretrain_batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=drop_last)

def prepare_cifar100_dataloaders(args):
    eval_transform = get_supervised_transform(args.image_size, is_train=False)
    train_transform = get_supervised_transform(args.image_size, is_train=True)
    cifar_root = os.path.join(args.dataset_root, 'cifar100')
    linear_classifier_train_dataset = torchvision.datasets.CIFAR100(root=cifar_root, train=True, transform=eval_transform, download=False)
    linear_classifier_test_dataset = torchvision.datasets.CIFAR100(root=cifar_root, train=False, transform=eval_transform, download=False)
    linear_classifier_train_loader = DataLoader(linear_classifier_train_dataset, batch_size=256, shuffle=False, num_workers=2)
    linear_classifier_test_loader = DataLoader(linear_classifier_test_dataset, batch_size=256, shuffle=False, num_workers=2)
    finetune_train_dataset = torchvision.datasets.CIFAR100(root=cifar_root, train=True, transform=train_transform, download=False)
    finetune_train_loader = DataLoader(finetune_train_dataset, batch_size=args.finetune_batch_size, shuffle=True, num_workers=2)
    return linear_classifier_train_loader, linear_classifier_test_loader, finetune_train_loader

# --- Model Architectures & Loss ---
class SimCLRModel(nn.Module):
    def __init__(self, backbone, projection_dim):
        super().__init__()
        self.backbone = backbone
        embedding_dim = self.backbone.config.hidden_size
        self.expansion_ratio = 4
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * self.expansion_ratio), 
            nn.ReLU(),
            nn.Linear(embedding_dim * self.expansion_ratio, projection_dim)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image tensor of shape [batch_size, 3, image_size, image_size]

        Returns:
            torch.Tensor: Output tensor after projection head of shape [batch_size, projection_dim]
        """
        h = self.backbone(pixel_values=x).last_hidden_state[:, 0, :]
        return self.projection_head(h)

class SupervisedModel(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)
    def forward(self, x):
        return self.classifier(self.backbone(pixel_values=x).last_hidden_state[:, 0, :])

class NTXentLoss(nn.Module):
    def __init__(self, device, batch_size, temperature):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        # TODO: Add any other variables needed
        self.cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.eps = 1e-8


    def forward(self, z_i, z_j):
        """
        Args:
            z_i (torch.Tensor): First set of representations
            z_j (torch.Tensor): Second set of representations
        Returns:
            torch.Tensor: Computed loss
        """
        # TODO: Implement the forward pass of NTXentLoss
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        N = z_i.shape[0]

        z = torch.cat([z_i, z_j], dim=0) 
        sim_matrix = torch.exp(torch.matmul(z, z.T) / self.temperature)

        mask = (~torch.eye(2*N, dtype=torch.bool, device=self.device)).float()
        masked_sim_matrix = sim_matrix * mask
        denominator = masked_sim_matrix.sum(dim=1)

        idx = torch.arange(N, device=self.device)
        sim_1 = sim_matrix[idx, idx+N]
        sim_2 = sim_matrix[idx+N, idx]
        numerator = torch.cat([sim_1, sim_2], dim=0)

        loss = -torch.log(numerator / denominator + self.eps)
        L = loss.mean()

        return L

# --- Training & Evaluation Functions ---
def get_model_backbone(args, pretrained=False):
    model_source = args.model_name_or_path
    if pretrained:
        print(f"Loading PRE-TRAINED backbone (Transformers) from: {model_source}")
        return AutoModel.from_pretrained(model_source)
    else:
        print(f"Loading RANDOMIZED backbone architecture (Transformers) from: {model_source}")
        config = AutoConfig.from_pretrained(model_source)
        return AutoModel.from_config(config)

def train_on_imagenet(args, writer, simclr=False, from_scratch=True):
    device = args.device
    start_state = "from Scratch" if from_scratch else "Fine-tuning DINO"
    method = "SimCLR" if simclr else "Supervised"
    print(f"\n--- Starting {method} Training on ImageNet ({start_state}) ---")

    backbone = get_model_backbone(args, pretrained=not from_scratch)
    train_loader = prepare_imagenet_loader(args, simclr=simclr)

    if simclr:
        model = SimCLRModel(backbone, args.projection_dim).to(device)
        criterion = NTXentLoss(device, args.pretrain_batch_size, args.temperature)
    else:
        model = SupervisedModel(backbone, args.num_pretrain_classes).to(device)
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=args.pretrain_lr)
    
    current_step = 0
    for epoch in range(args.pretrain_epochs):
        pbar_desc = f"{method} Epoch {epoch+1}"
        pbar = tqdm(train_loader, desc=pbar_desc, miniters=250, mininterval=25)
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        for data in pbar:
            optimizer.zero_grad()
            if simclr:
                view1, view2 = data[0].to(device), data[1].to(device)
                z1, z2 = model(view1), model(view2)
                loss = criterion(z1, z2)
            else:
                images, labels = data[0].to(device), data[1].to(device)
                predictions = model(images)
                loss = criterion(predictions, labels)
            
            loss.backward()
            optimizer.step()
            
            writer.add_scalar(f'Loss/pretrain_{method}', loss.item(), current_step)
            epoch_loss_sum += float(loss.item())
            epoch_loss_count += 1
            avg_loss = epoch_loss_sum / max(1, epoch_loss_count)
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Avg': f'{avg_loss:.4f}'}, refresh=False)
            
            current_step += 1
            if args.max_steps > 0 and current_step >= args.max_steps: break
        if args.max_steps > 0 and current_step >= args.max_steps:
            print(f"Reached max_steps ({args.max_steps}). Stopping.")
            break
    return model.backbone

def train_end_to_end_on_cifar(args, writer, finetune_train_loader, linear_classifier_test_loader, from_scratch=False):
    device = args.device
    start_state = "from Scratch" if from_scratch else "Fine-tuning DINO"
    print(f"\n--- Starting End-to-End Supervised Training on CIFAR-100 ({start_state}) ---")
    backbone = get_model_backbone(args, pretrained=not from_scratch)
    model = SupervisedModel(backbone, args.num_cifar_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.finetune_lr)
    criterion = nn.CrossEntropyLoss()
    
    current_step = 0
    for epoch in range(args.finetune_epochs):
        model.train()
        pbar = tqdm(finetune_train_loader, desc=f"CIFAR Train Epoch {epoch+1}", miniters=250, mininterval=25)
        for images, labels in pbar:
            optimizer.zero_grad()
            predictions = model(images.to(device))
            loss = criterion(predictions, labels.to(device))
            loss.backward()
            optimizer.step()
            writer.add_scalar('Loss/cifar_train', loss.item(), current_step)
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'}, refresh=False)
            current_step += 1
            if args.max_steps > 0 and current_step >= args.max_steps: break
        if args.max_steps > 0 and current_step >= args.max_steps:
            print(f"Reached max_steps ({args.max_steps}). Stopping.")
            break
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(linear_classifier_test_loader, desc="Final Evaluation", miniters=250, mininterval=25):
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Final CIFAR-100 End-to-End Accuracy: {accuracy:.2f}%")
    return accuracy

def evaluate_linear_head(args, writer, backbone, linear_classifier_train_loader, linear_classifier_test_loader):
    device = args.device
    print("\n--- Starting Linear Head Evaluation on CIFAR-100 ---")
    backbone.to(device).eval()

    @torch.no_grad()
    def extract_features(loader, desc):
        features, labels = [], []
        for images, lbls in tqdm(loader, desc=desc, miniters=10):
            feature = backbone(pixel_values=images.to(device)).last_hidden_state[:, 0, :]
            features.append(feature.cpu()); labels.append(lbls)
        return torch.cat(features), torch.cat(labels)

    train_features, train_labels = extract_features(linear_classifier_train_loader, "Extracting Train Features")
    test_features, test_labels = extract_features(linear_classifier_test_loader, "Extracting Test Features")

    classifier = nn.Linear(backbone.config.hidden_size, args.num_cifar_classes).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=args.linear_classifier_lr)
    criterion = nn.CrossEntropyLoss()
    feature_loader = DataLoader(torch.utils.data.TensorDataset(train_features, train_labels), batch_size=256, shuffle=True)

    print("Training the linear head...")
    for epoch in range(args.linear_classifier_epochs):
        for step, (features, labels) in enumerate(feature_loader):
            optimizer.zero_grad()
            predictions = classifier(features.to(device))
            loss = criterion(predictions, labels.to(device))
            loss.backward(); optimizer.step()
            writer.add_scalar('Loss/linear_linear_classifier', loss.item(), epoch * len(feature_loader) + step)

    print("Evaluating the linear head...")
    with torch.no_grad():
        predictions = classifier(test_features.to(device))
        correct = (predictions.argmax(dim=1).cpu() == test_labels).sum().item()
        accuracy = 100 * correct / len(test_labels)
    print(f"Linear Head Final Accuracy: {accuracy:.2f}%")
    return accuracy

def _build_run_dir(args):
    """Construct a readable run directory name.

    Pattern: YYYYmmdd-HHMMSS__<base_name>
    base_name = --run-name if provided else training_mode
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = (args.run_name or args.training_mode).replace(' ', '_')
    return os.path.join(args.output_dir, 'runs', f"{ts}__{base}")


def main(args):
    torch.manual_seed(42)
    np.random.seed(42)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {args.device}")
    run_dir = _build_run_dir(args)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Logging to: {run_dir}")
    writer = SummaryWriter(log_dir=run_dir)
    # Persist args for easy inspection
    with open(os.path.join(run_dir, 'args.json'), 'w') as f:
        json.dump({k: v for k, v in vars(args).items()}, f, indent=2)
    linear_classifier_train_loader, linear_classifier_test_loader, finetune_train_loader = prepare_cifar100_dataloaders(args)
    final_accuracy = 0.0

    # --- Mode Dispatcher ---
    mode = args.training_mode
    if mode == 'pretrained':
        backbone = get_model_backbone(args, pretrained=True)
        final_accuracy = evaluate_linear_head(args, writer, backbone, linear_classifier_train_loader, linear_classifier_test_loader)
    elif mode == 'scratch':
        backbone = get_model_backbone(args, pretrained=False)
        final_accuracy = evaluate_linear_head(args, writer, backbone, linear_classifier_train_loader, linear_classifier_test_loader)
    elif mode == 'simclr_scratch':
        backbone = train_on_imagenet(args, writer, simclr=True, from_scratch=True)
        final_accuracy = evaluate_linear_head(args, writer, backbone, linear_classifier_train_loader, linear_classifier_test_loader)
    elif mode == 'supervised_scratch':
        backbone = train_on_imagenet(args, writer, simclr=False, from_scratch=True)
        final_accuracy = evaluate_linear_head(args, writer, backbone, linear_classifier_train_loader, linear_classifier_test_loader)
    elif mode == 'simclr_pretrained':
        backbone = train_on_imagenet(args, writer, simclr=True, from_scratch=False)
        final_accuracy = evaluate_linear_head(args, writer, backbone, linear_classifier_train_loader, linear_classifier_test_loader)
    elif mode == 'supervised_pretrained':
        backbone = train_on_imagenet(args, writer, simclr=False, from_scratch=False)
        final_accuracy = evaluate_linear_head(args, writer, backbone, linear_classifier_train_loader, linear_classifier_test_loader)
    elif mode == 'cifar_supervised_pretrained':
        final_accuracy = train_end_to_end_on_cifar(args, writer, finetune_train_loader, linear_classifier_test_loader, from_scratch=False)
    elif mode == 'cifar_supervised_scratch':
        final_accuracy = train_end_to_end_on_cifar(args, writer, finetune_train_loader, linear_classifier_test_loader, from_scratch=True)
    
    # Log final results
    hparams = {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool))}
    writer.add_hparams(hparams, {'hparam/accuracy': final_accuracy}, run_name="hparams")
    writer.add_scalar('final/accuracy', final_accuracy, 0)
    writer.close()

    log_file = os.path.join(args.output_dir, 'results.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'a') as f:
        f.write(f"Mode: {args.training_mode}, Accuracy: {final_accuracy:.2f}%\n")

if __name__ == '__main__':
    print(f"Torch version: {torch.__version__}")
    parser = argparse.ArgumentParser(description="SimCLR and Supervised Training Framework")
    
    # Required arguments
    parser.add_argument('--training_mode', type=str, required=True, choices=[
        "pretrained",
        "scratch",
        "simclr_scratch",
        "supervised_scratch",
        "simclr_pretrained",
        "supervised_pretrained",
        "cifar_supervised_pretrained",
        "cifar_supervised_scratch"
    ], help="The experimental mode to run.")
    parser.add_argument('--run-name', type=str, default=None, help='Custom name for this run (defaults to training_mode).')
    
    # Paths (offline-friendly defaults)
    parser.add_argument('--dataset_root', type=str, default='./data', help="Root directory containing datasets (cifar100/, tiny-imagenet-200/ ...).")
    parser.add_argument('--imagenet_path', type=str, default='./data/tiny-imagenet-200', help="Path to tiny-imagenet-200 root containing train/ and val/ (used for all *IMAGENET* modes).")
    parser.add_argument('--output_dir', type=str, default='./outputs', help="Directory to save logs and results.")
    parser.add_argument('--model_name_or_path', type=str, default='./models/dinov3-vits16-pretrain-lvd1689m', help="Local path (downloaded by setup.py) or HF id for backbone model.")

    # General Training Hyperparameters
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--max_steps', type=int, default=1000, help="Max steps for training. -1 for full epochs.")
    
    # Pre-training Hyperparameters (pretraining on Imagenet)
    parser.add_argument('--pretrain_epochs', type=int, default=5)
    parser.add_argument('--pretrain_lr', type=float, default=3e-5)
    parser.add_argument('--pretrain_batch_size', type=int, default=32)
    parser.add_argument('--num_pretrain_classes', type=int, default=200, help='Number of classes for supervised pretraining (tiny-imagenet-200 has 200).')
    
    # Fine-tuning Hyperparameters (training end to end on CIFAR)
    parser.add_argument('--finetune_epochs', type=int, default=10)
    parser.add_argument('--finetune_lr', type=float, default=1e-4)
    parser.add_argument('--finetune_batch_size', type=int, default=32)
    
    # Linear Classifier Hyperparameters
    parser.add_argument('--linear_classifier_epochs', type=int, default=20)
    parser.add_argument('--linear_classifier_lr', type=float, default=1e-3)
    parser.add_argument('--num_cifar_classes', type=int, default=100)
    
    # SimCLR Hyperparameters
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--projection_dim', type=int, default=128)

    args = parser.parse_args()
    main(args)