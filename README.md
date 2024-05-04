To setup the project, run the following commands. You should have Python 3.10 installed already.

```
$ git submodule update --init --recursive
$ python3.10 -m venv venv
$ source venv/bin/activate
(venv) $ pip install -r requirements.txt --no-cache-dir
```

python -m ipykernel install --name=venv

(Note: for Windows, use 'wsl' to enter Linux before running 'source venv/bin/activate')
(Set kernel to installed 'venv' kernel in Jupyter notebook when running .ipynb files to access installed modules in requirements.txt)