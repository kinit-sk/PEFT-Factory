#!/bin/bash

# Copyright 2025 the PEFTFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#SBATCH -J "peft-factory-eval"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -o logs/peft-factory-eval-stdout.%J.out
#SBATCH -e logs/peft-factory-eval-stdout.%J.err
#SBATCH --time=2-00:00
#SBATCH --account=p1370-25-2

eval "$(conda shell.bash hook)"
conda activate peft-factory2
module load libsndfile

# datasets=(mnli qqp qnli sst2 stsb mrpc rte cola)
# peft_methods=(ia3 prompt-tuning lora lntuning)
# models=(gemma-3-1b-it llama-3-8b-instruct mistral-7b-instruct)

# datasets=(mnli qqp qnli sst2 stsb mrpc rte cola)
# peft_methods=(lora)
# models=(gemma-3-1b-it llama-3-8b-instruct mistral-7b-instruct)

export HF_HOME="/projects/${PROJECT}/cache"

# datasets=(mmlu piqa siqa hellaswag winogrande openbookqa math_qa gsm8k svamp conala codealpacapy apps)
# datasets=(mnli qqp qnli sst2 stsb mrpc rte cola record multirc boolq wic wsc cb copa mmlu piqa siqa hellaswag winogrande openbookqa math_qa gsm8k svamp conala codealpacapy apps)
datasets=(sst2)
peft_methods=(bitfit)
models=(llama-3-8b-instruct)


for d in ${datasets[@]};
do
    for m in ${models[@]};
    do
        for pm in ${peft_methods[@]};
        do
            saves=(saves_multiple/${pm}/${m}/train_${d}_*)

            TIMESTAMP=`date +%s`
            OUTPUT_DIR="saves_multiple/${pm}/${m}/eval_${d}_${TIMESTAMP}"
            ADAPTER="${saves[-1]}"
            DATASET="${d}_eval"
            SEED=123
            WANDB_PROJECT="peft-factory-eval-${pm}"
            WANDB_NAME="${pm}_${m}_eval_${d}"

            mkdir -p ${OUTPUT_DIR}

            export OUTPUT_DIR DATASET SEED ADAPTER WANDB_PROJECT WANDB_NAME

            envsubst < examples/peft/${pm}/${m}/eval.yaml > ${OUTPUT_DIR}/eval.yaml

            llamafactory-cli train ${OUTPUT_DIR}/eval.yaml

            python scripts/peftfactory/compute_metrics.py ${OUTPUT_DIR} ${d}
        done
    done
done
