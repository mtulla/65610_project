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
        self.c_attn1 = qnn.QuantLinear(config.n_embd, config.n_embd, **qlinear_args)
        self.c_attn2 = qnn.QuantLinear(config.n_embd, config.n_embd, **qlinear_args)
        self.c_attn3 = qnn.QuantLinear(config.n_embd, config.n_embd, **qlinear_args)
        # output projection
        self.c_proj = qnn.QuantLinear(config.n_embd, config.n_embd, **qlinear_args)
        # regularization
        # self.softmax_layer = torch.nn.Softmax(dim=-1)
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

    @staticmethod  
    def qsoftmax2(q_x: np.ndarray):
        """Compute the softmax function for ndarray.

        Args:
            q_x (ndarray): The input values.

        Returns:
            ndarray: The softmax outputs.
        """
        # Compute the max value for each sequence
        # q_x_max, _ = torch.max(q_x, axis=-1, keepdim=True)
        # q_x_max, _ = torch.max(q_x, axis=-1)
        # q_x_max = q_x_max.unsqueeze(-1)

        # Subtract max for numerical stability
        # q_x_minus_max = q_x - q_x_max

        # Apply the exponential
        x_exp = torch.exp(q_x)

        # Compute the sum along the sequence axis
        # q_x_exp_sum = torch.sum(x_exp, axis=-1, keepdims=True)
        print("x_exp shape", x_exp.shape)
        q_x_exp_sum = x_exp @ torch.ones_like(x_exp)

        # Compute the inverse of the sum
        # x_inverse_exp_sum = 1.0 / q_x_exp_sum
        # x_inverse_exp_sum = torch.ones_like(q_x_exp_sum) / q_x_exp_sum

        # Compute the final softmax values
        # q_x_softmax = x_exp * x_inverse_exp_sum
        q_x_softmax = x_exp/q_x_exp_sum

        return q_x_softmax


    def forward(self, x):
        x = self.quant_inp(x)
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        print("x size", x.size())
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn1(x), self.c_attn2(x), self.c_attn3(x)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        print("k size", k.size())
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        B, nh, T, hs = q.shape
        q = q.view(-1, T, hs)
        k = k.view(-1, T, hs)
        print("q shape", q.shape)
        print("k shape", k.shape)
        att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(hs))
        print("att size", att.size())
        print("type:" + str(att.type))
        print(att)
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) this one doesn't work, probably we have to change self.bias to be some array of floats with some -inf in it and then add it to att
        # att = self.softmax_layer(att) 
        print("att shape", att.size())
        att = QuantCasualSelfAttention.qsoftmax2(att)
        att = self.quant_inp(att) # quantize output from softmax
        att = self.attn_dropout(att)
        B, nh, T, hs = v.shape
        att = att.view(-1, T, T)
        v = v.view(-1, T, hs)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.view(B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y