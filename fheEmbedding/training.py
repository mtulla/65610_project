#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from ngram import QuantNGramLanguageModeler, pickle_to_path

N_BITS = 10
CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
 

def training_loop(model, ngrams, word_to_ix):
    losses = []
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    print(f"First 2 n-grams: {ngrams[:2]}")

    for epoch in range(10):
        total_loss = 0
        for context, target in ngrams:

            # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
            # into integer indices and wrap them in tensors)
            context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

            # Step 2. Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old
            # instance
            model.zero_grad()

            # Step 3. Run the forward pass, getting log probabilities over next
            # words
            out = model(context_idxs)
            log_probs = F.log_softmax(out, dim=1)

            # Step 4. Compute your loss function. (Again, Torch wants the target
            # word wrapped in a tensor)
            loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))

            # Step 5. Do the backward pass and update the gradient
            loss.backward()
            optimizer.step()

            # Get the Python number from a 1-element Tensor by calling tensor.item()
            total_loss += loss.item()
        
        losses.append(total_loss)

        print(f"Epoch: {epoch}, Loss: {total_loss}")

    print(f"Losses: {losses}")  # The loss decreased every iteration over the training data! 
    return losses


if __name__ == "__main__":
    with open("text.txt", "r") as f:
        text = f.read().split()

    # we should tokenize the input, but we will ignore that for now
    # build a list of tuples.
    # Each tuple is ([ word_i-CONTEXT_SIZE, ..., word_i-1 ], target word)
    ngrams = [
        (
            [text[i - j - 1] for j in range(CONTEXT_SIZE)],
            text[i]
        )
        for i in range(CONTEXT_SIZE, len(text))
    ]
    vocab = set(text)
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    vocab_size = len(vocab)

    model = QuantNGramLanguageModeler(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE)
    model.train()
    losses = training_loop(model, ngrams, word_to_ix)

    # Save the model and other parameters.
    params = (vocab_size, EMBEDDING_DIM, CONTEXT_SIZE)
    pickle_to_path(params, "model/params.pkl")
    pickle_to_path(word_to_ix, "model/word_to_ix.pkl")
    pickle_to_path(vocab, "model/vocab.pkl")
    pickle_to_path(losses, "model/losses.pkl")
    torch.save(model.state_dict(), "model/model.pth")
