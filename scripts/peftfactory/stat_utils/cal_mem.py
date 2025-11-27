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

import numpy as np
import pandas as pd
from tqdm import tqdm

import wandb


api = wandb.Api()

methods = ["lora", "prefix-tuning", "prompt-tuning", "p-tuning", "lntuning", "ia3"]

mean_memory_usage = {}

for m in methods:
    print(f"Processing method: {m}")
    project = f"rbelanec/peft-factory-train-{m}"
    runs = api.runs(project)

    max_values = []

    for run in tqdm(runs):
        history = run.history(stream="events")  # load just 1 row

        if history.empty or "system.gpu.0.memoryAllocatedBytes" not in history.columns:
            continue

        max_val = history["system.gpu.0.memoryAllocatedBytes"].max()

        max_values.append({"run": run.name.split("/")[-1], "id": run.id, "max_memory_usage": max_val})

    df = pd.DataFrame(max_values)
    mean_memory_usage[m] = np.round(df["max_memory_usage"].mean() / 1e9, 2)


df_mem = pd.DataFrame.from_dict(mean_memory_usage, orient="index", columns=["mean_memory_usage_GB"])
print(df_mem)
