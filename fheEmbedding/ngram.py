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

torch.manual_seed(1)

N_BITS = 10
CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
# We will use Shakespeare Sonnet 2
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()
# we should tokenize the input, but we will ignore that for now
# build a list of tuples.
# Each tuple is ([ word_i-CONTEXT_SIZE, ..., word_i-1 ], target word)
ngrams = [
    (
        [test_sentence[i - j - 1] for j in range(CONTEXT_SIZE)],
        test_sentence[i]
    )
    for i in range(CONTEXT_SIZE, len(test_sentence))
]
vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}

def word_to_tensor(word):
    return torch.nn.functional.one_hot(torch.tensor(word_to_ix[word]), len(vocab)).float()

def word_to_2d_tensor(word):
    return torch.nn.functional.one_hot(torch.tensor([word_to_ix[word]]), len(vocab)).float()


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


def training_loop(model):
    losses = []
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

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
    print(f"Loss: {losses}")  # The loss decreased every iteration over the training data! 

model = QuantNGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)

model.train()
training_loop(model)
model.eval()

print(f"Model: {model}")
inp = torch.tensor(word_to_ix["beauty"], dtype=torch.long)
print(f"Embedding of beauty: {model.embeddings(inp)}")

qembedding = qnn.QuantLinear(len(vocab), EMBEDDING_DIM, bias=False, bias_quant=None, input_quant=Int8ActPerTensorFloat)
qembedding.weight = nn.Parameter(model.embeddings.weight.transpose(0, 1))
inp = word_to_2d_tensor('beauty')
print(f"Quantized Embedding of beauty: {qembedding(inp)}")

# Compile the embedding.
input_data = torch.stack([word_to_tensor(word) for word in vocab])
input_data = np.array(input_data, dtype=float)
compiled_embedding = compile_brevitas_qat_model(
    qembedding,
    input_data,
)

# Run encrypted embedding
inp = np.array(word_to_2d_tensor("beauty"), dtype=float)
out = compiled_embedding.forward(inp, fhe="execute")
print(f"Embedding of beauty done in FHE {out}")