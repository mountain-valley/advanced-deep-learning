import torch
import torch.nn as nn
import torch.nn.functional as F
from src.components import Encoder, Decoder

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
        """
        Straight-through estimator for rounding operation.
        Args:
            x (Tensor): Input tensor.
        Returns:
            Tensor: Rounded tensor with gradient passthrough.
        The straight-through estimator allows gradients to pass through the rounding operation
        unchanged during backpropagation, effectively treating the rounding as an identity function
        in the backward pass. This is useful in scenarios where we want to quantize values during
        the forward pass but still allow gradient-based optimization to occur as if the rounding
        did not happen.
        """
        return (x.round() - x).detach() + x

    def quantize(self, z):
        """
        Quantize the input feature map z to discrete levels using straight-through estimator for rounding.
        Args:
            z (Tensor): Input feature map of shape (B, dim, H, W).
        Returns:
            z_q (Tensor): Quantized feature map of shape (B, dim, H, W).
        Instructions:
            1. Apply tanh to z to constrain values between -1 and 1.
            2. Scale z by levels/2 to map to quantization levels.
            3. Use straight-through estimator to round scaled values to nearest integer and
                rescale rounded values back to original range.
            4. Clamp the quantized values to ensure they remain within [-1, 1].
        """
        # TODO: Implement the quantization process as described in the instructions.
        pass

    def forward(self, x):
        """
        Forward pass through the FSQ model.
        Args:
            x (Tensor): Input tensor of shape (B, 3, H, W).
        Returns:
            reconstructed_x (Tensor): Reconstructed output tensor of shape (B, 3, H, W).
            total_loss (Tensor): Combined reconstruction and commitment loss.
            (num_tokens, levels, None): Tuple containing number of tokens, levels, and None for indices.
        Instructions:
            1. Encode x to get feature map
            2. Quantize using FSQ layer
            3. Decode
            4. Compute reconstruction and commitment losses
        """
        # TODO: Implement the forward pass as described in the instructions.
        ###################################
        pass
        ###################################
        # return x_hat, reconstruction_loss + commitment_loss, (24*24, self.levels, None)
