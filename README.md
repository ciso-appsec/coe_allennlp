
# coe_allennlp

CoE mirror of supported AllenNLP models

### How To build:

Note: In the steps below, I've cloned the coe-allennlp and coe_allennlp-models GitHub repos in root/publicgithub/coe_allennlp/ and root/publicgithub/coe_allennlp-models/. Update the paths if you clone the repos to a different folder.

Note: You may want to use proxy repo to Docker Hub instead of pulling directly from Docker Hub, to avoid the Docker Hub rate limit. Just use `Your-proxy-repo/python:3.9.16` as the docker image in the commands below instead of just `python:3.9.16`.

Note: The `checklist` package was not installed in the pre-CoE default allennlp python build, but `checklist` was included in the pre-CoE allennlp Docker images. Right now, there's this issue with installing `checklist`, so `pip install allennlp[checklist]` and `pip install checklist==0.0.11` both fail: https://github.com/marcotcr/checklist/issues/144. We haven't yet found a solution, so the CoE docker images are currently missing the `checklist` package.

- allennlp python build to install the package from local source code:
```
docker run --rm -v /root/publicgithub:/root/publicgithub -it python:3.9.16 bash

# In the docker container:
cd /root/publicgithub/coe_allennlp
rm -rf my-virtual-env
python -m venv my-virtual-env
source my-virtual-env/bin/activate
pip install -U pip setuptools wheel
# Use .[dev,all] instead of .[all] to install dev dependencies also (for unit tests, etc).
pip install --editable .[all]
allennlp test-install

# Still in the docker container, do a Mend scan:
echo userKey=YOUR_USER_KEY >> wss-unified-agent.config
echo apiKey=YOUR_API_KEY >> wss-unified-agent.config
echo wss.url=YOUR_MEND_SERVER_URL/agent >> wss-unified-agent.config
echo productName=YOUR_MEND_PRODUCT_NAME >> wss-unified-agent.config
echo projectName=coeAllenNLP >> wss-unified-agent.config
apt-get update
apt install default-jre
curl -OL https://unified-agent.s3.amazonaws.com/wss-unified-agent.jar
# Be careful if you create files named like temp-requirements.txt, Mend will include them in the scan.
# Make sure Mend scans only the released dependencies, not the dev-requirements:
mv dev-requirements.txt dev-requirements
# Optionally, deactivate your virtual env and install virtualenv so Mend can succesfully do `pip install` as part of the scan.
# If you don't do this, Mend will still scan your requirements.txt file. You may want to do `pip freeze > pipfreeze.log; diff requirements.txt pipfreeze.log`
# to check that all dependencies are included in requirements.txt.
deactivate
pip install virtualenv --user
java -jar wss-unified-agent.jar
mv dev-requirements dev-requirements.txt
```

- allennlp docker build:
```
docker build -f Dockerfile -t coe-allennlp/coe-allennlp:latest .

# Run a small test:
docker run --rm coe-allennlp/coe-allennlp:latest test-install
```

- allennlp run unit tests:
```
git checkout 80fb6061e568cb9d6ab5d45b661e86eb61b92c82 # The code before the CoE updates
--OR--
git checkout main # Or checkout whatever code you want to test


docker run --rm --entrypoint bash -it -v /root/publicgithub:/root/publicgithub allennlp/allennlp:latest # Last release from AllenAI
--OR--
docker build -f Dockerfile -t coe-allennlp/coe-allennlp:latest . && docker run --rm --entrypoint bash -it -v /root/publicgithub:/root/publicgithub coe-allennlp/coe-allennlp:latest # Use the docker image you built yourself

# Inside the docker container:
cd /root/publicgithub/coe_allennlp
pip install -r dev-requirements.txt
# The checklist feature requires additional non-default dependencies to be installed, so skip those tests.
# One of the tests in f1_measure_test.py seems to get stuck, or is extremely slow, so skip that.
# tests/training/metrics/fbeta_measure_test.py also gets stuck quite often. These issues with getting stuck may be due to CoE updates?
pytest --ignore-glob='*checklist*' --ignore='tests/training/metrics/f1_measure_test.py'
# If you want to run just one file, run `pytest path/to/the/file.py`

# With the code before the CoE updates and the last  docker image released from AllenAI, only the following two tests fail. (Perhaps this is because we're running as root, so we always have 'write' permissions even when we're not supposed to?):
# FAILED tests/common/file_utils_test.py::TestFileLock::test_locking - Failed: DID NOT RAISE <class 'PermissionError'>
# FAILED tests/common/file_utils_test.py::TestTensorCache::test_tensor_cache - assert False
```

- allennlp-models python build to create the python wheel (.whl) file:
```
docker run --rm -v /root/publicgithub:/root/publicgithub -it python:3.9.16 bash -c 'cd root/publicgithub/coe_allennlp-models/ ; python setup.py bdist_wheel'

# Do a Mend scan:
docker run --rm -v /root/publicgithub:/root/publicgithub -it python:3.9.16 bash
# The following commands are run inside the Docker container:
cd root/publicgithub/coe_allennlp-models/
echo userKey=YOUR_USER_KEY >> wss-unified-agent.config
echo apiKey=YOUR_API_KEY >> wss-unified-agent.config
echo wss.url=YOUR_MEND_SERVER_URL/agent >> wss-unified-agent.config
echo productName=YOUR_MEND_PRODUCT_NAME >> wss-unified-agent.config
echo projectName=coeAllenNLPModels >> wss-unified-agent.config
apt-get update
apt install default-jre
curl -OL https://unified-agent.s3.amazonaws.com/wss-unified-agent.jar
# Be careful if you create files named like temp-requirements.txt, Mend will include them in the scan.
# Make sure Mend scans only the released dependencies, not the dev-requirements:
mv dev-requirements.txt dev-requirements
pip install virtualenv --user
java -jar wss-unified-agent.jar
mv dev-requirements dev-requirements.txt
```

- allennlp-models docker build:
    - Note: You must complete the allennlp docker build first, for the base image.
    - Note: You must complete the allennlp-models python build first, for the wheel (.whl) file.
```
docker build --no-cache --progress=plain --build-arg ALLENNLP_IMAGE=coe-allennlp/coe-allennlp --build-arg ALLENNLP_TAG=latest -f Dockerfile -t coe-allennlp/coe-allennlp-models:latest .

# Test that installing allennlp-models didn't break the allennlp installation:
docker run --rm coe-allennlp/coe-allennlp-models:latest test-install
```

- allennlp-models run unit tests
    - Note: check out the commit you want to test, and build the allennlp-models docker image first. Or, use the `allennlp/models:latest` docker image from dockerio for the last official build from the original AllenNLP poject.
```
docker run --rm --entrypoint bash -it -v /root/publicgithub:/root/publicgithub coe-allennlp/coe-allennlp-models:latest

# In the docker container:

cd /root/publicgithub/coe_allennlp-models/
pip install -r dev-requirements.txt
# The checklist feature requires additional non-default dependencies to be installed, so skip those tests.
# The Makefile has a `-m "not pretrained_model_test and not pretrained_config_test"` option, it seems to be because those tests are very slow.
# tests/classification/interpret/sst_test.py got stuck for me too or is just slow (on my retry, it passed after 2 minutes), so skip that.
pytest --ignore-glob='*checklist*' -m "not pretrained_model_test and not pretrained_config_test" --ignore=tests/classification/interpret/sst_test.py
# All tests from the above command should pass.
```

## Mend Docker image scanning

Note that this scan doesn't seem to find any python dependency vulnerabilities.
```
curl https://downloads.mend.io/cli/linux_amd64/mend -o mend && chmod +x mend

./mend auth login
# Mend environment: Other → [YOUR_MEND_SERVER_URL]
# Product: [x]  SCA (Dependencies) / CN (Image)
# User Email: [YOUR_EMAIL]
# User Key: [YOUR KEY From the User Profile page in the Mend UI]

./mend image [YOUR DOCKER IMAGE, for example: coe-allennlp/coe-allennlp-models:latest]
```


# Original README.md

<div align="center">
    <br>
    <img src="https://raw.githubusercontent.com/allenai/allennlp/main/docs/img/allennlp-logo-dark.png" width="400"/>
    <p>
    An Apache 2.0 NLP research library, built on PyTorch, for developing state-of-the-art deep learning models on a wide variety of linguistic tasks.
    </p>
    <hr/>
</div>
<p align="center">
    <a href="https://github.com/allenai/allennlp/actions">
        <img alt="CI" src="https://github.com/allenai/allennlp/workflows/CI/badge.svg?event=push&branch=main">
    </a>
    <a href="https://pypi.org/project/allennlp/">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/allennlp">
    </a>
    <a href="https://github.com/allenai/allennlp/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/allenai/allennlp.svg?color=blue&cachedrop">
    </a>
    <a href="https://codecov.io/gh/allenai/allennlp">
        <img alt="Codecov" src="https://codecov.io/gh/allenai/allennlp/branch/main/graph/badge.svg">
    </a>
    <a href="https://optuna.org">
        <img alt="Optuna" src="https://img.shields.io/badge/Optuna-integrated-blue">
    </a>
    <br/>
</p>

⚠️ **NOTICE:** The AllenNLP library is now in maintenance mode. That means we are no longer adding new features or upgrading dependencies. We will still respond to questions and address bugs as they arise up until December 16th, 2022. If you have any concerns or are interested in maintaining AllenNLP going forward, please open an issue on this repository.

AllenNLP has been a big success, but as the field is advancing quickly it's time to focus on new initiatives. We're working hard to make [AI2 Tango](https://github.com/allenai/tango) the best way to organize research codebases. If you are an active user of AllenNLP, here are some suggested alternatives:
* If you like the trainer, the configuration language, or are simply looking for a better way to manage your experiments, check out [AI2 Tango](https://github.com/allenai/tango).
* If you like AllenNLP's `modules` and `nn` packages, check out [delmaksym/allennlp-light](https://github.com/delmaksym/allennlp-light). It's even compatible with [AI2 Tango](https://github.com/allenai/tango)!
* If you like the framework aspect of AllenNLP, check out [flair](https://github.com/flairNLP/flair). It has multiple state-of-art NLP models and allows you to easily use pretrained embeddings such as those from transformers.
* If you like the AllenNLP metrics package, check out [torchmetrics](https://torchmetrics.readthedocs.io/en/stable/). It has the same API as AllenNLP, so it should be a quick learning curve to make the switch.
* If you want to vectorize text, try [the transformers library](https://github.com/huggingface/transformers).
* If you want to maintain the AllenNLP Fairness or Interpret components, please get in touch. There is no alternative to it, so we are looking for a dedicated maintainer.
* If you are concerned about other AllenNLP functionality, please create an issue. Maybe we can find another way to continue supporting your use case.

## Quick Links

- ↗️ [Website](https://allennlp.org/)
- 🔦 [Guide](https://guide.allennlp.org/)
- 🖼 [Gallery](https://gallery.allennlp.org)
- 💻 [Demo](https://demo.allennlp.org)
- 📓 [Documentation](https://docs.allennlp.org/) ( [latest](https://docs.allennlp.org/latest/) | [stable](https://docs.allennlp.org/stable/) | [commit](https://docs.allennlp.org/main/) )
- ⬆️ [Upgrade Guide from 1.x to 2.0](https://github.com/allenai/allennlp/discussions/4933)
- ❓ [Stack Overflow](https://stackoverflow.com/questions/tagged/allennlp)
- ✋ [Contributing Guidelines](CONTRIBUTING.md)
- 🤖 [Officially Supported Models](https://github.com/allenai/allennlp-models)
    - [Pretrained Models](https://github.com/allenai/allennlp-models/blob/main/allennlp_models/pretrained.py)
    - [Documentation](https://docs.allennlp.org/models/) ( [latest](https://docs.allennlp.org/models/latest/) | [stable](https://docs.allennlp.org/models/stable/) | [commit](https://docs.allennlp.org/models/main/) )
- ⚙️ [Continuous Build](https://github.com/allenai/allennlp/actions)
- 🌙 [Nightly Releases](https://pypi.org/project/allennlp/#history)

## In this README

- [Getting Started Using the Library](#getting-started-using-the-library)
- [Plugins](#plugins)
- [Package Overview](#package-overview)
- [Installation](#installation)
    - [Installing via pip](#installing-via-pip)
    - [Installing using Docker](#installing-using-docker)
    - [Installing from source](#installing-from-source)
- [Running AllenNLP](#running-allennlp)
- [Issues](#issues)
- [Contributions](#contributions)
- [Citing](#citing)
- [Team](#team)

## Getting Started Using the Library

If you're interested in using AllenNLP for model development, we recommend you check out the
[AllenNLP Guide](https://guide.allennlp.org) for a thorough introduction to the library, followed by our more advanced guides
on [GitHub Discussions](https://github.com/allenai/allennlp/discussions/categories/guides).

When you're ready to start your project, we've created a couple of template repositories that you can use as a starting place:

* If you want to use `allennlp train` and config files to specify experiments, use [this
  template](https://github.com/allenai/allennlp-template-config-files). We recommend this approach.
* If you'd prefer to use python code to configure your experiments and run your training loop, use
  [this template](https://github.com/allenai/allennlp-template-python-script). There are a few
  things that are currently a little harder in this setup (loading a saved model, and using
  distributed training), but otherwise it's functionality equivalent to the config files
  setup.

In addition, there are external tutorials:

* [Hyperparameter optimization for AllenNLP using Optuna](https://medium.com/optuna/hyperparameter-optimization-for-allennlp-using-optuna-54b4bfecd78b)
* [Training with multiple GPUs in AllenNLP](https://medium.com/ai2-blog/tutorial-how-to-train-with-multiple-gpus-in-allennlp-c4d7c17eb6d6)
* [Training on larger batches with less memory in AllenNLP](https://medium.com/ai2-blog/tutorial-training-on-larger-batches-with-less-memory-in-allennlp-1cd2047d92ad)
* [How to upload transformer weights and tokenizers from AllenNLP to HuggingFace](https://medium.com/ai2-blog/tutorial-how-to-upload-transformer-weights-and-tokenizers-from-allennlp-to-huggingface-ecf6c0249bf)

And others on the [AI2 AllenNLP blog](https://medium.com/ai2-blog/allennlp/home).

## Plugins

AllenNLP supports loading "plugins" dynamically. A plugin is just a Python package that
provides custom registered classes or additional `allennlp` subcommands.

There is ecosystem of open source plugins, some of which are maintained by the AllenNLP
team here at AI2, and some of which are maintained by the broader community.

<table>
<tr>
    <td><b> Plugin </b></td>
    <td><b> Maintainer </b></td>
    <td><b> CLI </b></td>
    <td><b> Description </b></td>
</tr>
<tr>
    <td> <a href="https://github.com/allenai/allennlp-models"><b>allennlp-models</b></a> </td>
    <td> AI2 </td>
    <td> No </td>
    <td> A collection of state-of-the-art models </td>
</tr>
<tr>
    <td> <a href="https://github.com/allenai/allennlp-semparse"><b>allennlp-semparse</b></a> </td>
    <td> AI2 </td>
    <td> No </td>
    <td> A framework for building semantic parsers </td>
</tr>
<tr>
    <td> <a href="https://github.com/allenai/allennlp-server"><b>allennlp-server</b></a> </td>
    <td> AI2 </td>
    <td> Yes </td>
    <td> A simple demo server for serving models </td>
</tr>
<tr>
    <td> <a href="https://github.com/himkt/allennlp-optuna"><b>allennlp-optuna</b></a> </td>
    <td> <a href="https://himkt.github.io/profile/">Makoto Hiramatsu</a> </td>
    <td> Yes </td>
    <td> <a href="https://optuna.org/">Optuna</a> integration for hyperparameter optimization </td>
</tr>
</table>

AllenNLP will automatically find any official AI2-maintained plugins that you have installed,
but for AllenNLP to find personal or third-party plugins you've installed,
you also have to create either a local plugins file named `.allennlp_plugins`
in the directory where you run the `allennlp` command, or a global plugins file at `~/.allennlp/plugins`.
The file should list the plugin modules that you want to be loaded, one per line.

To test that your plugins can be found and imported by AllenNLP, you can run the `allennlp test-install` command.
Each discovered plugin will be logged to the terminal.

For more information about plugins, see the [plugins API docs](https://docs.allennlp.org/main/api/common/plugins/). And for information on how to create a custom subcommand
to distribute as a plugin, see the [subcommand API docs](https://docs.allennlp.org/main/api/commands/subcommand/).

## Package Overview

<table>
<tr>
    <td><b> allennlp </b></td>
    <td> An open-source NLP research library, built on PyTorch </td>
</tr>
<tr>
    <td><b> allennlp.commands </b></td>
    <td> Functionality for the CLI </td>
</tr>
<tr>
    <td><b> allennlp.common </b></td>
    <td> Utility modules that are used across the library </td>
</tr>
<tr>
    <td><b> allennlp.data </b></td>
    <td> A data processing module for loading datasets and encoding strings as integers for representation in matrices </td>
</tr>
<tr>
    <td><b> allennlp.fairness </b></td>
    <td> A module for bias mitigation and fairness algorithms and metrics </td>
</tr>
<tr>
    <td><b> allennlp.modules </b></td>
    <td> A collection of PyTorch modules for use with text </td>
</tr>
<tr>
    <td><b> allennlp.nn </b></td>
    <td> Tensor utility functions, such as initializers and activation functions </td>
</tr>
<tr>
    <td><b> allennlp.training </b></td>
    <td> Functionality for training models </td>
</tr>
</table>

## Installation

AllenNLP requires Python 3.6.1 or later and [PyTorch](https://pytorch.org/).

We support AllenNLP on Mac and Linux environments. We presently do not support Windows but are open to contributions.

### Installing via conda-forge

The simplest way to install AllenNLP is using conda (you can choose a different python version):

```
conda install -c conda-forge python=3.8 allennlp
```

To install optional packages, such as `checklist`, use

```
conda install -c conda-forge allennlp-checklist
```

or simply install `allennlp-all` directly. The plugins mentioned above are similarly installable, e.g.

```
conda install -c conda-forge allennlp-models allennlp-semparse allennlp-server allennlp-optuna
```

### Installing via pip

It's recommended that you install the PyTorch ecosystem **before** installing AllenNLP by following the instructions on [pytorch.org](https://pytorch.org/).

After that, just run `pip install allennlp`.



> ⚠️ If you're using Python 3.7 or greater, you should ensure that you don't have the PyPI version of `dataclasses` installed after running the above command, as this could cause issues on certain platforms. You can quickly check this by running `pip freeze | grep dataclasses`. If you see something like `dataclasses=0.6` in the output, then just run `pip uninstall -y dataclasses`.

If you need pointers on setting up an appropriate Python environment or would like to install AllenNLP using a different method, see below.

#### Setting up a virtual environment

[Conda](https://conda.io/) can be used set up a virtual environment with the
version of Python required for AllenNLP.  If you already have a Python 3
environment you want to use, you can skip to the 'installing via pip' section.

1.  [Download and install Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

2.  Create a Conda environment with Python 3.8 (3.7 or 3.9 would work as well):

    ```
    conda create -n allennlp_env python=3.8
    ```

3.  Activate the Conda environment. You will need to activate the Conda environment in each terminal in which you want to use AllenNLP:

    ```
    conda activate allennlp_env
    ```

#### Installing the library and dependencies

Installing the library and dependencies is simple using `pip`.

```bash
pip install allennlp
```

To install the optional dependencies, such as `checklist`, run

```bash
pip install allennlp[checklist]
```
Or you can just install all optional dependencies with `pip install allennlp[all]`.

*Looking for bleeding edge features? You can install nightly releases directly from [pypi](https://pypi.org/project/allennlp/#history)*

AllenNLP installs a script when you install the python package, so you can run allennlp commands just by typing `allennlp` into a terminal.  For example, you can now test your installation with `allennlp test-install`.

You may also want to install `allennlp-models`, which contains the NLP constructs to train and run our officially
supported models, many of which are hosted at [https://demo.allennlp.org](https://demo.allennlp.org).

```bash
pip install allennlp-models
```

### Installing using Docker

Docker provides a virtual machine with everything set up to run AllenNLP--
whether you will leverage a GPU or just run on a CPU.  Docker provides more
isolation and consistency, and also makes it easy to distribute your
environment to a compute cluster.

AllenNLP provides [official Docker images](https://hub.docker.com/r/allennlp/allennlp) with the library and all of its dependencies installed.

Once you have [installed Docker](https://docs.docker.com/engine/installation/),
you should also install the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
if you have GPUs available.

Then run the following command to get an environment that will run on GPU:

```bash
mkdir -p $HOME/.allennlp/
docker run --rm --gpus all -v $HOME/.allennlp:/root/.allennlp allennlp/allennlp:latest
```

You can test the Docker environment with

```bash
docker run --rm --gpus all -v $HOME/.allennlp:/root/.allennlp allennlp/allennlp:latest test-install 
```

If you don't have GPUs available, just omit the `--gpus all` flag.

#### Building your own Docker image

For various reasons you may need to create your own AllenNLP Docker image, such as if you need a different version
of PyTorch. To do so, just run `make docker-image` from the root of your local clone of AllenNLP.

By default this builds an image with the tag `allennlp/allennlp`, but you can change this to anything you want
by setting the `DOCKER_IMAGE_NAME` flag when you call `make`. For example,
`make docker-image DOCKER_IMAGE_NAME=my-allennlp`.

If you want to use a different version of Python or PyTorch, set the flags `DOCKER_PYTHON_VERSION` and `DOCKER_TORCH_VERSION` to something like
`3.9` and `1.9.0-cuda10.2`, respectively. These flags together determine the base image that is used. You can see the list of valid
combinations in this GitHub Container Registry: [github.com/allenai/docker-images/pkgs/container/pytorch](https://github.com/allenai/docker-images/pkgs/container/pytorch).

After building the image you should be able to see it listed by running `docker images allennlp`.

```
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
allennlp/allennlp   latest              b66aee6cb593        5 minutes ago       2.38GB
```

### Installing from source

You can also install AllenNLP by cloning our git repository:

```bash
git clone https://github.com/allenai/allennlp.git
```

Create a Python 3.7 or 3.8 virtual environment, and install AllenNLP in `editable` mode by running:

```bash
pip install -U pip setuptools wheel
pip install --editable .[dev,all]
```

This will make `allennlp` available on your system but it will use the sources from the local clone
you made of the source repository.

You can test your installation with `allennlp test-install`.
See [https://github.com/allenai/allennlp-models](https://github.com/allenai/allennlp-models)
for instructions on installing `allennlp-models` from source.

## Running AllenNLP

Once you've installed AllenNLP, you can run the command-line interface
with the `allennlp` command (whether you installed from `pip` or from source).
`allennlp` has various subcommands such as `train`, `evaluate`, and `predict`.
To see the full usage information, run `allennlp --help`.

You can test your installation by running  `allennlp test-install`.

## Issues

Everyone is welcome to file issues with either feature requests, bug reports, or general questions.  As a small team with our own internal goals, we may ask for contributions if a prompt fix doesn't fit into our roadmap.  To keep things tidy we will often close issues we think are answered, but don't hesitate to follow up if further discussion is needed.

## Contributions

The AllenNLP team at AI2 ([@allenai](https://github.com/allenai)) welcomes contributions from the community. 
If you're a first time contributor, we recommend you start by reading our [CONTRIBUTING.md](https://github.com/allenai/allennlp/blob/main/CONTRIBUTING.md) guide.
Then have a look at our issues with the tag [**`Good First Issue`**](https://github.com/allenai/allennlp/issues?q=is%3Aissue+is%3Aopen+label%3A%22Good+First+Issue%22).

If you would like to contribute a larger feature, we recommend first creating an issue with a proposed design for discussion. This will prevent you from spending significant time on an implementation which has a technical limitation someone could have pointed out early on. Small contributions can be made directly in a pull request.

Pull requests (PRs) must have one approving review and no requested changes before they are merged.  As AllenNLP is primarily driven by AI2 we reserve the right to reject or revert contributions that we don't think are good additions.

## Citing

If you use AllenNLP in your research, please cite [AllenNLP: A Deep Semantic Natural Language Processing Platform](https://www.semanticscholar.org/paper/AllenNLP%3A-A-Deep-Semantic-Natural-Language-Platform-Gardner-Grus/a5502187140cdd98d76ae711973dbcdaf1fef46d).

```bibtex
@inproceedings{Gardner2017AllenNLP,
  title={AllenNLP: A Deep Semantic Natural Language Processing Platform},
  author={Matt Gardner and Joel Grus and Mark Neumann and Oyvind Tafjord
    and Pradeep Dasigi and Nelson F. Liu and Matthew Peters and
    Michael Schmitz and Luke S. Zettlemoyer},
  year={2017},
  Eprint = {arXiv:1803.07640},
}
```

## Team

AllenNLP is an open-source project backed by [the Allen Institute for Artificial Intelligence (AI2)](https://allenai.org/).
AI2 is a non-profit institute with the mission to contribute to humanity through high-impact AI research and engineering.
To learn more about who specifically contributed to this codebase, see [our contributors](https://github.com/allenai/allennlp/graphs/contributors) page.
