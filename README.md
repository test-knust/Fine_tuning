🧩 Installing Python Packages
Prerequisites

Python 3.13 or later is required.

Recommended: create and use a virtual environment to keep dependencies isolated.

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






## Installing python packages Using pip

Once the environment is activated, upgrade pip and install all dependencies from requirements.txt:

```bash

pip install --upgrade pip
pip install -r requirements.txt
```

## This will install all necessary libraries for fine-tuning and running LLMs, including:

```bash

PyTorch
Transformers
Datasets
Accelerate
PEFT (LoRA fine-tuning)
```
