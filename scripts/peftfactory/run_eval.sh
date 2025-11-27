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

# datasets=(mnli qqp qnli sst2 stsb mrpc rte cola)
# peft_methods=(ia3 prefix-tuning prompt-tuning lora lntuning)
# models=(gemma-3-1b-it llama-3-8b-instruct mistral-7b-instruct)

datasets=(sst2)
# datasets=(mnli qqp qnli sst2 stsb mrpc rte cola)
peft_methods=(bitfit)
# peft_methods=(base)
models=(llama-3-8b-instruct)

for d in ${datasets[@]};
do
    for m in ${models[@]};
    do
        for pm in ${peft_methods[@]};
        do
            # saves=(saves/${pm}/${m}/train_${d}_*)
            saves=(saves_multiple/${pm}/${m}/train_${d}_*)
            
            TIMESTAMP=`date +%s`
            # OUTPUT_DIR="saves/${pm}/${m}/eval_${d}_${TIMESTAMP}"
            OUTPUT_DIR="saves_multiple/${pm}/${m}/eval_${d}_${TIMESTAMP}"

            if [[ "${pm}" == "base" ]]; then
                ADAPTER=""
            else
                ADAPTER="${saves[-1]}"
            fi

            if [[ "${pm}" == *"adapter"* ]]; then
                ADAPTER="${ADAPTER}/${d}"
            fi
        
            DATASET="${d}_eval"
            SEED=123
            WANDB_PROJECT="peft-factory-eval-${pm}"

            mkdir -p ${OUTPUT_DIR}

            export OUTPUT_DIR DATASET SEED ADAPTER WANDB_PROJECT

            envsubst < examples/peft/${pm}/${m}/eval.yaml > ${OUTPUT_DIR}/eval.yaml

            llamafactory-cli train ${OUTPUT_DIR}/eval.yaml

            python scripts/peftfactory/compute_metrics.py ${OUTPUT_DIR} ${d}
        done
    done
done
