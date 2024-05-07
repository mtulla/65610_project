#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.ao.quantization
import torch.ao.nn
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat
from concrete.ml.torch.compile import compile_brevitas_qat_model
import numpy as np
from pathlib import Path
import pickle
import string


def pickle_to_path(obj, path):
    p = Path(path)
    with p.open("wb") as f:
        pickle.dump(obj, f)

def pickle_from_path(path):
    p = Path(path)
    with p.open("rb") as f:
        return pickle.load(f)

def word_to_tensor(word, word_to_ix, vocab_size):
    return torch.nn.functional.one_hot(torch.tensor(word_to_ix[word]), vocab_size).float()

def word_to_2d_tensor(word, word_to_ix, vocab_size):
    return torch.nn.functional.one_hot(torch.tensor([word_to_ix[word]]), vocab_size).float()

def tokenize(text):
    for punct in string.punctuation:
        text = text.replace(punct, " " + punct + " ")
    return text.split()


class QuantNGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(QuantNGramLanguageModeler, self).__init__()
        self.embeddings = qnn.QuantEmbedding(vocab_size, embedding_dim)
        self.linear1 = qnn.QuantLinear(context_size * embedding_dim, 128, bias=True, bias_quant=None)
        self.linear2 = qnn.QuantLinear(128, vocab_size, bias=True, bias_quant=None)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = torch.relu(self.linear1(embeds))
        out = self.linear2(out)
        return out

class QuantLinearEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(QuantLinearEmbedding, self).__init__()
        self.quant_inp = qnn.QuantIdentity(return_quant_tensor=True)
        self.embeddings = qnn.QuantLinear(vocab_size, embedding_dim, bias=False, bias_quant=None)

    def forward(self, inputs):
        quant_inp = self.quant_inp(inputs)
        embeds = self.embeddings(quant_inp)
        return embeds
