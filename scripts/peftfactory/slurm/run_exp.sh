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

# datasets=(cb copa wsc svamp conala rte mrpc openbookqa wic stsb cola gsm8k siqa math_qa winogrande sst2 hellaswag qnli)
# datasets=(copa conala rte mrpc openbookqa wic stsb cola gsm8k siqa math_qa winogrande sst2 hellaswag qnli mnli mmlu qqp apps codealpacapy boolq piqa record multirc)
# peft_methods=(prefix-tuning prompt-tuning p-tuning lora lntuning ia3)
datasets=(cb copa svamp cola sst2 hellaswag wsc)
peft_methods=(bitfit)
models=(llama-3-8b-instruct)
# seeds=(42 123 456 789 101112)
seeds=(123 456 789 101112)
# EPOCHS=20
EPOCHS=10

for s in ${seeds[@]};
do
    for d in ${datasets[@]};
    do
        for m in ${models[@]};
        do
            for pm in ${peft_methods[@]};
            do
                TIMESTAMP=`date +%s`
                OUTPUT_DIR="saves_multiple/${pm}/${m}/train_${d}_${s}_${TIMESTAMP}"
                DATASET="${d}"
                SEED="${s}"
                WANDB_PROJECT="peft-factory-multiple-${pm}"
                WANDB_NAME="${pm}_${m}_train_${d}_${s}_${TIMESTAMP}"

                mkdir -p ${OUTPUT_DIR}

                export OUTPUT_DIR DATASET SEED WANDB_PROJECT WANDB_NAME EPOCHS
                envsubst < examples/peft/${pm}/${m}/train.yaml > ${OUTPUT_DIR}/train.yaml

                OUTPUT_DIR="saves_multiple/${pm}/${m}/eval_${d}_${s}_${TIMESTAMP}"
                WANDB_NAME="${pm}_${m}_eval_${d}_${s}_${TIMESTAMP}"
                ADAPTER="saves_multiple/${pm}/${m}/train_${d}_${s}_${TIMESTAMP}"
                DATASET="${d}_eval"

                mkdir -p ${OUTPUT_DIR}

                export OUTPUT_DIR WANDB_NAME ADAPTER DATASET
                envsubst < examples/peft/${pm}/${m}/eval.yaml > ${OUTPUT_DIR}/eval.yaml

                sbatch --job-name ${pm}_${m}_multiple_${d}_${s}_${TIMESTAMP} -o logs/${pm}_${m}_multiple_${d}_${s}_${TIMESTAMP}.out -e logs/${pm}_${m}_multiple_${d}_${s}_${TIMESTAMP}.err scripts/peftfactory/slurm/run_train_eval.sh saves_multiple/${pm}/${m}/train_${d}_${s}_${TIMESTAMP}/train.yaml saves_multiple/${pm}/${m}/eval_${d}_${s}_${TIMESTAMP}/eval.yaml saves_multiple/${pm}/${m}/eval_${d}_${s}_${TIMESTAMP} ${d}

                sleep 1
            done
        done
    done
done