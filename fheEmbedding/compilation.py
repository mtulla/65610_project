#!/usr/bin/env python3

import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat
from concrete.ml.torch.compile import compile_brevitas_qat_model
from concrete.ml.common.serialization.dumpers import dump
from pathlib import Path
import numpy as np

from ngram import (
    QuantNGramLanguageModeler,
    word_to_tensor,
    pickle_to_path,
    pickle_from_path,
)


if __name__ == "__main__":
    # Load model and other variables.
    print("Importing model and other variables...")
    vocab_size, embedding_dim, context_size = pickle_from_path("model/params.pkl")
    model = QuantNGramLanguageModeler(vocab_size, embedding_dim, context_size)
    model.load_state_dict(torch.load("model/model.pth"))
    model.eval()
    word_to_ix = pickle_from_path("model/word_to_ix.pkl")
    vocab = pickle_from_path("model/vocab.pkl")

    # Linearize the embedding.
    print("Linearizing the embedding...")
    qembedding = qnn.QuantLinear(vocab_size, embedding_dim, bias=False, bias_quant=None, input_quant=Int8ActPerTensorFloat)
    qembedding.weight = nn.Parameter(model.embeddings.weight.transpose(0, 1))

    # Compile the embedding.
    print("Compiling the embedding...")
    input_data = torch.stack([word_to_tensor(word, word_to_ix, vocab_size) for word in vocab])
    input_data = np.array(input_data, dtype=float)
    compiled_embedding = compile_brevitas_qat_model(
        qembedding,
        input_data,
    )

    # Save the compiled embedding.
    print("Saving the compiled embedding...")
    save_path = Path("model/compiled_embedding.pth")
    with save_path.open("w") as f:
        dump(compiled_embedding, f)
