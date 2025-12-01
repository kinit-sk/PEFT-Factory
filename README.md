![# PEFT Factory](assets/logo.png)


<div align="center" markdown="1">

----

## Parameter-Efficient Fine-Tuning made easy

PEFT-Factory is a fork of [LLaMa-Factory](https://github.com/hiyouga/LLaMA-Factory) ❤️, upgraded with easy to use **PEFT interface**, support for **HuggingFace PEFT methods** and **datasets** for benchmarking PEFT.
</div>


---
<div align="center" markdown="1">

## Supported methods

| PEFT method name            | Support |
|-----------------------------|---------|
| LoRA (including variants)   | ✅ 🦙  |
| OFT                         | ✅ 🦙  |
| Prefix Tuning               | ✅ 🤗  |
| Prompt Tuning               | ✅ 🤗  |
| P-Tuning                    | ✅ 🤗  |
| P-Tuning v2                 | ✅ 🤗  |
| MPT                         | ✅ 🤗  |
| IA3                         | ✅ 🤗  |
| LNTuning                    | ✅ 🤗  |
| Bottleneck Adapter          | ✅ 🤖  |
| Parallal Adapter            | ✅ 🤖  |
| SeqBottleneck Adapter       | ✅ 🤖  |
| SVFT                        | ✅ ⚙️  |
| BitFit                      | ✅ ⚙️  |


</div>

# Usage

This section provides instructions on how to install PEFT-Factory, download necessary data and methods, and run training using both command line and web UI.

## Quickstart

For video example please visit the [PEFT-Factory Demonstration
Video](https://www.youtube.com/watch?v=Q3kxvlyO-XY).

``` bash
# install package
pip install peftfactory

# dowload repo that contains data, PEFT methods and examples
git clone https://github.com/kinit-sk/PEFT-Factory.git && cd PEFT-Factory

# start web UI
pf webui
```

Alternatively, you can run training from command line:

``` bash
# install package
pip install peftfactory

# dowload repo that contains data, PEFT methods and examples
git clone https://github.com/kinit-sk/PEFT-Factory.git && cd PEFT-Factory
```

### Create some variables for envsubst

``` bash
# run training with config file
TIMESTAMP=`date +%s`
OUTPUT_DIR="saves/bitfit/llama-3.2-1b-instruct/train_wsc_${TIMESTAMP}"
DATASET="wsc"
SEED=123
WANDB_PROJECT="peft-factory-train-bitfit"
WANDB_NAME="bitfit_llama-3.2-1b-instruct_train_wsc"

mkdir -p "${OUTPUT_DIR}"

export OUTPUT_DIR DATASET SEED WANDB_PROJECT WANDB_NAME
```

### Use the template

Utility `envsubst` replaces the occurances of env variables with their values (see the template).

```bash
envsubst < examples/peft/bitfit/llama-3.2-1b-instruct/train.yaml > ${OUTPUT_DIR}/train.yaml
```

### Run the factory

```bash
peftfactory-cli train ${OUTPUT_DIR}/train.yaml
```

## Installation

There are multiple ways to install PEFT-Factory. You can install
develelopment version from source or install the latest release from PyPI.

### Using pip

```bash
pip install peftfactory
```

### From Source

#### Clone the repository

```bash
git clone git@github.com:kinit-sk/PEFT-Factory.git
```

#### Build the wheel package

```bash
make build
```

#### Install with pip

```bash
pip install dist/[name of the built package].whl
```

## Get data and methods

To download data, methods and examples for training please download the
repository from GitHub.

```bash
git clone https://github.com/kinit-sk/PEFT-Factory.git && cd PEFT-Factory
```

## Run training

You can run training from command line or using web UI.

### From Command Line

To run training from command line use the following command:

```bash
pf train [path to config file].yaml
```

### Using web UI

To run the web UI use the following command:

```bash
pf webui
```
