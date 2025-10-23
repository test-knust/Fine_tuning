

# Installation

## Create a Virtual Environment (optional)

We recommend using a virtual environment for the installation. This will ensure that the python dependencies for the workshop are isolated from the rest of your system.



## Create and activate for Linux/Mac::

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Windows command-line:
```batch
# The command may be `python3` instead of `python` depending on your setup
python -m venv .venv
.venv\Scripts\activate.bat
```

## To deactivate later, run:

```bash
deactivate
```






## Install Using pip

Install the `autogen-agentchat` package using pip:

```bash

pip install -U "autogen-agentchat"
```

```{note}
Python 3.10 or later is required.
```

## Install OpenAI for Model Client

To use the OpenAI and Azure OpenAI models, you need to install the following
extensions:

```bash
pip install "autogen-ext[openai]"
```

If you are using Azure OpenAI with AAD authentication, you need to install the following:

```bash
pip install "autogen-ext[azure]"

