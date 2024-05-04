To setup the project, run the following commands. You should have Python 3.10 installed already.

```
$ git submodule update --init --recursive
$ python3.10 -m venv venv
$ source venv/bin/activate
(venv) $ pip install -r requirements.txt --no-cache-dir
```

Before running the jupyter notebook, run python -m ipykernel install --name=venv

