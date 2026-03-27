![# PEFT Factory](assets/logo.png)

<div align="center" markdown="1">

----

### Parameter-Efficient Fine-Tuning Made Easy

PEFT-Factory is a fork of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) ❤️, enhanced with an easy-to-use **PEFT interface**, support for **HuggingFace PEFT methods**, and curated **datasets** for benchmarking PEFT approaches.

📄 [**System Demonstration Paper**](https://aclanthology.org/2026.eacl-demo.15.pdf) &nbsp;|&nbsp; 🎥 [**Demo Video**](https://www.youtube.com/watch?v=Q3kxvlyO-XY) &nbsp;|&nbsp; 🏛️ [**EACL 2026**](#citation)

### 🏆 **PEFT-Factory was presented at EACL 2026** (19th Conference of the European Chapter of the Association for Computational Linguistics, Rabat, Morocco) as a System Demonstration.

</div>

---

## Supported Methods

<div align="center" markdown="1">

| PEFT Method                 | Supported | Backend |
|-----------------------------|:---------:|---------|
| LoRA (including variants)   | ✅ | 🦙 LLaMA-Factory |
| OFT                         | ✅ | 🦙 LLaMA-Factory |
| Prefix Tuning               | ✅ | 🤗 HuggingFace PEFT |
| Prompt Tuning               | ✅ | 🤗 HuggingFace PEFT |
| P-Tuning                    | ✅ | 🤗 HuggingFace PEFT |
| P-Tuning v2                 | ✅ | 🤗 HuggingFace PEFT |
| MPT                         | ✅ | 🤗 HuggingFace PEFT |
| IA³                         | ✅ | 🤗 HuggingFace PEFT |
| LNTuning                    | ✅ | 🤗 HuggingFace PEFT |
| Bottleneck Adapter          | ✅ | 🤖 AdapterHub |
| Parallel Adapter            | ✅ | 🤖 AdapterHub |
| SeqBottleneck Adapter       | ✅ | 🤖 AdapterHub |
| SVFT                        | ✅ | ⚙️ Custom |
| BitFit                      | ✅ | ⚙️ Custom |

</div>

---

## Usage

This section provides instructions on how to install PEFT-Factory, download the necessary data and methods, and run training using either the command line or the web UI.

## Quickstart

For a video walkthrough, visit the [PEFT-Factory Demonstration Video](https://www.youtube.com/watch?v=Q3kxvlyO-XY).

```bash
# Install the package
pip install peftfactory

# Download the repository, which contains data, PEFT methods, and examples
git clone https://github.com/kinit-sk/PEFT-Factory.git && cd PEFT-Factory

# Start the web UI
pf webui
```

Alternatively, you can run training directly from the command line:

```bash
# Install the package
pip install peftfactory

# Download the repository, which contains data, PEFT methods, and examples
git clone https://github.com/kinit-sk/PEFT-Factory.git && cd PEFT-Factory
```

### Set Environment Variables for `envsubst`

Define the variables that will be substituted into the training config template:

```bash
TIMESTAMP=`date +%s`
OUTPUT_DIR="saves/bitfit/llama-3.2-1b-instruct/train_wsc_${TIMESTAMP}"
DATASET="wsc"
SEED=123
WANDB_PROJECT="peft-factory-train-bitfit"
WANDB_NAME="bitfit_llama-3.2-1b-instruct_train_wsc"

mkdir -p "${OUTPUT_DIR}"

export OUTPUT_DIR DATASET SEED WANDB_PROJECT WANDB_NAME
```

### Apply the Config Template

The `envsubst` utility replaces occurrences of environment variables in the template file with their current values:

```bash
envsubst < examples/peft/bitfit/llama-3.2-1b-instruct/train.yaml > ${OUTPUT_DIR}/train.yaml
```

### Run Training

```bash
peftfactory-cli train ${OUTPUT_DIR}/train.yaml
```

---

## Installation

PEFT-Factory can be installed in several ways: directly from PyPI for the latest release, or built from source for the development version.

### From PyPI (Recommended)

```bash
pip install peftfactory
```

### From Source

**1. Clone the repository:**

```bash
git clone git@github.com:kinit-sk/PEFT-Factory.git
```

**2. Build the wheel package:**

```bash
make build
```

**3. Install with pip:**

```bash
pip install dist/[name of the built package].whl
```

### Installing DeepSpeed

DeepSpeed is required for evaluation and computation of the PSCP metric.

```bash
pip install deepspeed
```

> **Note:** You may encounter an error about the `CUDA_HOME` environment variable not being set. The fix depends on your environment:

#### Conda

```bash
conda install -c nvidia cuda-compiler
```

#### Standard virtualenv / pyenv

You will need to install CUDA with the `nvcc` compiler at the OS level. Instructions vary by operating system — consult your distribution's documentation. For example, on Arch Linux:

```bash
# Arch Linux example — the exact command differs per OS
sudo pacman -S cuda
```

---

## Data and Methods

To download the datasets, PEFT method implementations, and example configs for training, clone the repository from GitHub:

```bash
git clone https://github.com/kinit-sk/PEFT-Factory.git && cd PEFT-Factory
```

---

## Running Training

### From the Command Line

```bash
pf train [path to config file].yaml
```

### Using the Web UI

```bash
pf webui
```

---

## Citation

If you use PEFT-Factory in your research, please cite our EACL 2026 System Demonstration paper:

```bibtex
@inproceedings{belanec-etal-2026-peft-factory,
    title = "{PEFT}-Factory: Unified Parameter-Efficient Fine-Tuning of Autoregressive Large Language Models",
    author = "Belanec, Robert  and
      Srba, Ivan  and
      Bielikova, Maria",
    editor = "Croce, Danilo  and
      Leidner, Jochen  and
      Moosavi, Nafise Sadat",
    booktitle = "Proceedings of the 19th Conference of the {E}uropean Chapter of the {A}ssociation for {C}omputational {L}inguistics (Volume 3: System Demonstrations)",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.eacl-demo.15/",
    doi = "10.18653/v1/2026.eacl-demo.15",
    pages = "188--202",
    ISBN = "979-8-89176-382-1",
    abstract = "Parameter-Efficient Fine-Tuning (PEFT) methods address the increasing size of Large Language Models (LLMs). Currently, many newly introduced PEFT methods are challenging to replicate, deploy, or compare with one another. To address this, we introduce PEFT-Factory, a unified framework for efficient fine-tuning LLMs using both off-the-shelf and custom PEFT methods. While its modular design supports extensibility, it natively provides a representative set of 19 PEFT methods, 27 classification and text generation datasets addressing 12 tasks, and both standard and PEFT-specific evaluation metrics. As a result, PEFT-Factory provides a ready-to-use, controlled, and stable environment, improving replicability and benchmarking of PEFT methods. PEFT-Factory is a downstream framework that originates from the popular LLaMA-Factory, and is publicly available at https://github.com/kinit-sk/PEFT-Factory."
}
```