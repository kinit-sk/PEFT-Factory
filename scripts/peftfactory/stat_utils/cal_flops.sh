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

CONFIG_DIR="scripts/peftfactory/stat_utils/configs"
SCRIPT="scripts/peftfactory/stat_utils/cal_flops.py"
RESULTS=()

for config in "$CONFIG_DIR"/*.yaml; do
    method=$(basename "$config" .yaml)
    echo "Running $SCRIPT with method: $method"
    python "$SCRIPT" "$config"
    result_file="scripts/peftfactory/stat_utils/flops_${method}.json"
    if [[ -f "$result_file" ]]; then
        number=$(jq '.number' "$result_file")
        echo "FLOPs for $method: $number"
        RESULTS+=("$method: $number")
    fi
done

echo "Summary of all FLOPs numbers:"
for entry in "${RESULTS[@]}"; do
    echo "$entry"
done