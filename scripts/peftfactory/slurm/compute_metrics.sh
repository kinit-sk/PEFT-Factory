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

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -o logs/peft-factory-cm-stdout.%J.out
#SBATCH -e logs/peft-factory-cm-stderr.%J.out
#SBATCH --time=2-00:00
#SBATCH --account=p1370-25-2

eval "$(conda shell.bash hook)"
conda activate peft-factory
module load libsndfile

export HF_HOME="/projects/${PROJECT}/cache"

# datasets=(mnli qqp qnli sst2 stsb mrpc rte cola record multirc boolq wic wsc cb copa)
# datasets=(record multirc boolq wic wsc cb copa)
datasets=(mnli cb)
peft_methods=(base)
models=(llama-3-8b-instruct)


for d in ${datasets[@]};
do
    for m in ${models[@]};
    do
        for pm in ${peft_methods[@]};
        do
            saves=(saves/${pm}/${m}/eval_${d}_*)

            EVAL_DIR="${saves[-1]}"

            python scripts/peftfactory/compute_metrics.py ${EVAL_DIR} ${d}
        done
    done
done
