#!/usr/bin/env python3

import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat
from concrete.ml.common.serialization.loaders import load
import numpy as np
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

from ngram import (
    QuantNGramLanguageModeler,
    word_to_2d_tensor,
    word_to_tensor,
    pickle_from_path,
)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename="logs/evaluation.log",
        level=logging.INFO
    )

    # Load model, FHE circuit, and other variables.
    logger.info("Loading model, FHE circuit, and other variables...")
    vocab_size, embedding_dim, context_size = pickle_from_path("model/params.pkl")
    model = QuantNGramLanguageModeler(vocab_size, embedding_dim, context_size)
    model.load_state_dict(torch.load("model/model.pth"))
    model.eval()
    word_to_ix = pickle_from_path("model/word_to_ix.pkl")
    vocab = pickle_from_path("model/vocab.pkl")
    circuit_path = Path("model/compiled_embedding.pth")
    with circuit_path.open("r") as f:
        compiled_embedding = load(f)

    # Compile the embedding.
    logger.info("Compiling the embedding...")
    input_data = torch.stack([word_to_tensor(word, word_to_ix, vocab_size) for word in vocab])
    input_data = np.array(input_data, dtype=float)
    compiled_embedding.compile(input_data)

    # Show embedding from model.
    inp = torch.tensor(word_to_ix["beauty"], dtype=torch.long)
    logger.info(f"Embedding of beauty: {model.embeddings(inp)}")

    # Show embedding after linearizing.
    qembedding = qnn.QuantLinear(vocab_size, embedding_dim, bias=False, bias_quant=None, input_quant=Int8ActPerTensorFloat)
    qembedding.weight = nn.Parameter(model.embeddings.weight.transpose(0, 1))
    inp = word_to_2d_tensor("beauty", word_to_ix, vocab_size)
    logger.info(f"Linearized embedding of beauty: {qembedding(inp)}")

    # Run encrypted embedding.
    inp = np.array(word_to_2d_tensor("beauty", word_to_ix, vocab_size), dtype=float)
    out = compiled_embedding.forward(inp)
    logger.info(f"Embedding of beauty done in FHE {out}")
