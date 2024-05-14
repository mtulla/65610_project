#!/usr/bin/env python3

import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat
from concrete.ml.common.serialization.loaders import load
import numpy as np
from pathlib import Path
import logging
import time
import random
logger = logging.getLogger(__name__)

from ngram import (
    QuantNGramLanguageModeler,
    word_to_2d_tensor,
    word_to_tensor,
    pickle_from_path,
)

N_WORDS = 1000 # Number of words to profile

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
    compilation_time0 = time.perf_counter()
    compiled_embedding.compile(input_data)
    compilation_time1 = time.perf_counter()
    compilation_time = compilation_time1 - compilation_time0

    
    words = list(word_to_ix.keys())
    random.shuffle(words)
    words_to_embed = words[:N_WORDS]

    # Get embedding from model.
    total_embedding_time = 0
    embeddings = {}
    for word in words_to_embed:
        inp = torch.tensor(word_to_ix[word], dtype=torch.long)
        embedding_time0 = time.perf_counter()
        emb = model.embeddings(inp)
        embedding_time1 = time.perf_counter()
        logger.info(f"Embedding of {word}: {emb}")
        embeddings[word] = emb.detach().numpy()

        total_embedding_time += embedding_time1 - embedding_time0

    # Run encrypted embedding.
    total_fhe_time = 0
    fhe_embeddings = {}
    for word in words_to_embed:
        inp = np.array(word_to_2d_tensor(word, word_to_ix, vocab_size), dtype=float)
        embedding_time0 = time.perf_counter()
        emb = compiled_embedding.forward(inp)
        embedding_time1 = time.perf_counter()
        logger.info(f"Embedding of {word} done in FHE {emb}")
        fhe_embeddings[word] = emb

        total_fhe_time += embedding_time1 - embedding_time0

    avg_error = 0
    for word in words_to_embed:
        emb = embeddings[word]
        fhe_emb = fhe_embeddings[word]
        error = np.linalg.norm(emb - fhe_emb) / np.linalg.norm(emb)
        avg_error += error
    avg_error /= N_WORDS

    logger.info(f"Compilation time: {compilation_time}")
    logger.info(f"Total embedding time: {total_embedding_time}")
    logger.info(f"Total FHE embedding time: {total_fhe_time}")
    logger.info(f"Average error: {avg_error}")
