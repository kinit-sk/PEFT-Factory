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

import sys

import wandb


def get_max_gpu_memory(run_id, entity=None, project=None):
    """Fetch the maximum GPU memory allocation (in bytes) from a wandb run.

    Looks for the 'system.gpu.memoryAllocated' metric.
    """
    api = wandb.Api()
    if entity and project:
        run = api.run(f"{entity}/{project}/{run_id}")
    else:
        run = api.run(run_id)
    history = run.history(samples=10000)
    if "system.gpu.memoryAllocated" in history.columns:
        max_mem = history["system.gpu.memoryAllocated"].max()
        print(f"Max GPU memory allocated for run {run_id}: {max_mem} bytes")
        return max_mem
    else:
        print("No GPU memory allocation data found for this run.")
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_wandb_stats.py <run_id> [entity] [project]")
        sys.exit(1)
    run_id = sys.argv[1]
    entity = sys.argv[2] if len(sys.argv) > 2 else None
    project = sys.argv[3] if len(sys.argv) > 3 else None
