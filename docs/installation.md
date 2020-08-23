The latest release of `formulaic` is always published to the Python
Package Index (PyPI), from which it is available to download @
[https://pypi.org/project/formulaic/](https://pypi.org/project/formulaic/).

If your Python environment is provisioned with `pip`, installing `formulaic`
from the PyPI is as simple as running:

```
$ pip install formulaic
```

Otherwise, simply download the latest source package, extract it, and run from
within the top-level "formulaic" source directory:

```
$ python setup.py install
```

!!! note
    If you have a non-standard setup, ensure that `pip` and/or `python` above
    are replaced with the executables corresponding to the environment for which
    you are interested in installing `formulaic`. This is done automatically if
    you are using a virtual environment.

You are ready to use Formulaic. If you would like some guidance, please refer
to the [Basic Usage](basic/intro.md) and [Advanced Usage](advanced/intro.md)
guides.


## Installing for development

If you are interested in developing `formulaic`, you should clone the source
code repository, and install in editable mode from there (allowing your changes
to be instantly available to all new Python sessions).

To clone the source code, run:

```
$ git clone git@github.com:matthewwardrop/formulaic.git
```

!!! note
    This requires you to have a GitHub account set up. If you do not have an
    account you can replace the SSH url above with
    `https://github.com/matthewwardrop/formulaic.git`. Also, if you are planning
    to submit your work upstream, you may wish to fork the repository into your
    own namespace first, and clone from there.

To install in editable mode, run:
```
$ pip install -e <path_to_cloned_formulaic_repo>
```

You can then make any changes you like to the repo, and have them be reflected
in your local Python sessions. Happy hacking, and I look forward to your
contributions!
