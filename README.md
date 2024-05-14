To setup the project, run the following commands. You should have Python 3.10 installed already.

```
$ git submodule update --init --recursive
$ python3.10 -m venv venv
$ source venv/bin/activate
(venv) $ pip install -r requirements.txt --no-cache-dir
```

The `fheEmbedding/` directory contains the relevant code to the project.
- `training.py` can be used to train the n-gram language model.
- `compilation.py` will attempt to compile and FHE circuit implementation of the word embedding layer and save it.
- `evaluation.py` will compute word embeddings for 1000 arbitraty tokens/words and log the results of the embdedings as well as the performance of the computations and their accuracy.