from brevitas import nn as qnn
from brevitas.core.quant import QuantType
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat
import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np

N_BITS = 3

class QuantCasualSelfAttention(nn.Module):
    def __init__(
            self,
            config,
            qlinear_args={
                "weight_bit_width": N_BITS,
                "weight_quant": Int8WeightPerTensorFloat,
                "bias": True,
                "bias_quant": None,
                "narrow_range": True,
            },
            qidentity_args={
                "bit_width": N_BITS,
                "act_quant": Int8ActPerTensorFloat,
            }
    ):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.quant_inp = qnn.QuantIdentity(**qidentity_args)
        # key, query, value projections for all heads, but in a batch
        self.c_attn = qnn.QuantLinear(config.n_embd, 3 * config.n_embd, **qlinear_args)
        # output projection
        self.c_proj = qnn.QuantLinear(config.n_embd, config.n_embd, **qlinear_args)
        # regularization
        self.attn_dropout = qnn.QuantDropout(config.attn_pdrop)
        self.resid_dropout = qnn.QuantDropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    @staticmethod
    def qsoftmax(q_x: np.ndarray):
        """Compute the softmax function, with quantized values.

        Args:
            q_x (DualArray): The quantized values to consider.

        Returns:
            q_x_softmax (DualArray): The quantized outputs.
        """
        # Compute the max value for each sequence
        q_x_max = np.max(q_x, axis=q_x.shape[-1] - 1, keepdims=True)

        # Subtract max for numerical stability
        q_x_minus_max = q_x - q_x_max

        # Apply the exponential
        x_exp = np.exp(q_x_minus_max)

        # Compute the sum along the sequence axis
        q_x_exp_sum = np.sum(x_exp, axis=1, keepdims=True)

        # Compute the inverse of the sum
        x_inverse_exp_sum = 1/q_x_exp_sum

        # Compute the final softmax values
        q_x_softmax = x_exp * x_inverse_exp_sum

        return q_x_softmax

    def forward(self, x):
        x = self.quant_inp(x)
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y