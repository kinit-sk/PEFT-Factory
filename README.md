![# PEFT Factory](assets/logo.png)


<div align="center" markdown="1">

----

### Parameter-Efficient Fine-Tuning made easy

PEFT-Factory is a fork of [LLaMa-Factory](https://github.com/hiyouga/LLaMA-Factory) ❤️, upgraded with easy to use **PEFT interface**, support for **HuggingFace PEFT methods** and **datasets** for benchmarking PEFT.
</div>


### Installation
⚠️ Currently, the best way is to install from this repository. This will change soon.

#### Clone the repository

```bash
git clone git@github.com:Wicwik/PEFT-Factory.git
```

#### Build the wheel
```bash
make build
```

#### Install wiht pip
```bash
pip install dist/llamafactory-0.9.4.dev0-py3-none-any.whl
```

### Quickstart
Quick start of prefix tuning with LLaMa3-8B-Instruct for CB dataset. We are using the template from examples directory, but feel free to use your own `config.yml`.


#### Create some variables for `envsubst`

```bash
TIMESTAMP=`date +%s`
OUTPUT_DIR="saves/prefix-tuning/llama-3-8b-instruct/train_cb_${TIMESTAMP}"
DATASET="cb"
SEED=123
WANDB_PROJECT="peft-factory-train-prefix-tuning"
WANDB_NAME="prefix-tuning_llama-3-8b-instruct_train_cb"

mkdir -p "${OUTPUT_DIR}"

export OUTPUT_DIR DATASET SEED WANDB_PROJECT WANDB_NAME
```

#### Use the template
Utility `envsubst` replaces the occurances of env variables with their values (see the [template](https://github.com/Wicwik/PEFT-Factory/blob/main/examples/peft/prefix-tuning/llama-3-8b-instruct/train.yaml)).

```bash
envsubst < examples/peft/prefix-tuning/llama-3-8b-instruct/train.yaml > ${OUTPUT_DIR}/train.yaml
```

#### Run the factory
```bash
llamafactory-cli train ${OUTPUT_DIR}/train.yaml
```

---
<div align="center" markdown="1">

### Supported methods

| PEFT method name           | Support |
| -------------------------- | ------- |
| LoRA (including variants)  | ✅ 🦙  |
| Prefix Tuning              | ✅ 🤗  |
| Prompt Tuning              | ✅ 🤗  |
| P-Tuning                   | ✅ 🤗  |
| LNTuning                   | ✅ 🤗  |
| SVD                        | ✅ ⚙️  |
| BitFit                     | ✅ ⚙️  |


</div>