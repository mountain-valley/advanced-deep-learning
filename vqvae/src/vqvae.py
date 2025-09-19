import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize
from src.components import Encoder, Decoder
from src.my_vector_quantizer import MyVectorQuantizer

class VQVAE(nn.Module):
    """
    VQ-VAE using shared Encoder/Decoder, with middle: permute/reshape -> VQ -> reshape/permute.
    """
    def __init__(self, dim=128, codebook_size=1024, custom=False):
        super().__init__()
        self.enc = Encoder(dim)
        self.dec = Decoder(dim)
        if custom:
            self.vq = MyVectorQuantizer(dim=dim, codebook_size=codebook_size, decay=0.99, commitment_weight=0.25)
        else:
            self.vq = VectorQuantize(dim=dim, codebook_size=codebook_size, decay=0.99, commitment_weight=0.25)

    def forward(self, x):
        """
        Forward pass through the VQ-VAE.
        Args:
            x (Tensor): Input tensor of shape (B, 3, H, W).
        Returns:
            reconstructed_x (Tensor): Reconstructed output tensor of shape (B, 3, H, W).
            total_loss (Tensor): Combined reconstruction and commitment loss.
            (num_tokens, codebook_size, indices): Tuple containing number of tokens, codebook size, and quantization indices.
        Instructions:
            1. Encode x to get feature map
            2. Reshape and permute feature map for VQ layer
            3. Quantize using VQ layer
            4. Reshape and permute quantized output back to feature map shape
            5. Decode
        """
        # TODO: Implement the forward pass as described in the instructions.
        ###################################
        pass
        
        ###################################
        # return x_hat, reconstruction_loss + commit_loss, (H*W, self.vq.codebook_size, indices)
