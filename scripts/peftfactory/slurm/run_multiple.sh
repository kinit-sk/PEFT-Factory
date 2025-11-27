#!/bin/bash

# Copyright 2025 the PEFT-Factory team.
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

# GLUE
# datasets=(mnli qqp qnli sst2 stsb mrpc rte cola)
# peft_methods=(ia3 prompt-tuning lora lntuning)
# models=(gemma-3-1b-it llama-3-8b-instruct mistral-7b-instruct)

# SuperGLUE
# datasets=(record multirc boolq wic wsc cb copa)
# peft_methods=(ia3 prompt-tuning lora lntuning)
# models=(gemma-3-1b-it llama-3-8b-instruct mistral-7b-instruct)

# datasets=(mnli qqp qnli sst2 stsb mrpc rte cola record multirc boolq wic wsc cb copa)
# datasets=(mmlu piqa siqa hellaswag winogrande openbookqa math_qa gsm8k svamp conala codealpacapy apps)
# datasets=(mnli qqp qnli sst2 stsb mrpc rte cola record multirc boolq wic wsc cb copa mmlu piqa siqa hellaswag winogrande openbookqa math_qa gsm8k svamp conala codealpacapy apps)
# datasets=(mnli qqp qnli sst2 stsb mrpc rte cola record multirc boolq wic wsc cb copa mmlu piqa siqa hellaswag winogrande openbookqa math_qa gsm8k svamp conala codealpacapy apps)
datasets=(qnli multirc)
peft_methods=(prefix-tuning)
models=(llama-3-8b-instruct)

for d in ${datasets[@]};
do
    for m in ${models[@]};
    do
        for pm in ${peft_methods[@]};
        do
            TIMESTAMP=`date +%s`
            OUTPUT_DIR="saves/${pm}/${m}/train_${d}_${TIMESTAMP}"
            DATASET="${d}"
            SEED=123
            WANDB_PROJECT="peft-factory-train-${pm}"
            WANDB_NAME="${pm}_${m}_train_${d}"

            mkdir -p ${OUTPUT_DIR}

            export OUTPUT_DIR DATASET SEED WANDB_PROJECT WANDB_NAME

            envsubst < examples/peft/${pm}/${m}/train.yaml > ${OUTPUT_DIR}/train.yaml

            sbatch --job-name ${pm}_${m}_train_${d}_${TIMESTAMP} -o logs/${pm}_${m}_train_${d}_${TIMESTAMP}.out -e logs/${pm}_${m}_train_${d}_${TIMESTAMP}.err scripts/peftfactory/slurm/run_single.sh ${OUTPUT_DIR}/train.yaml
            
            sleep 1
        done
    done
done
