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
#
# Copyright 2025 Microsoft Corporation and the LlamaFactory team.
#
# This code is inspired by the Microsoft's DeepSpeed library.
# https://www.deepspeed.ai/tutorials/flops-profiler/
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

import json

import fire
import torch
from deepspeed.accelerator import get_accelerator  # type: ignore
from deepspeed.profiling.flops_profiler import get_model_profile  # type: ignore

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.hparams import get_train_args, read_args
from llamafactory.model import load_model, load_tokenizer


def save_number_to_json(number, filename):
    """Save a single number to a JSON file."""
    with open(filename, "w") as f:
        json.dump({"number": number}, f)


def calculate_flops(
    path_to_config=None,
):
    r"""Calculate the flops of pre-trained models.

    Usage: python cal_flops.py path_to_config --batch_size 1 --seq_length 512
    """
    args = None
    args = read_args(args)

    batch_size: int = 1
    seq_length: int = 512

    model_args, data_args, training_args, finetuning_args, _, peft_args = get_train_args(args)

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    _ = get_template_and_fix_tokenizer(tokenizer, data_args)

    with get_accelerator().device(0):
        model = load_model(tokenizer, model_args, finetuning_args, peft_args, training_args.do_train)

        fake_input = torch.ones((batch_size, seq_length), dtype=torch.long, device=model.device)
        input_dict = {"input_ids": fake_input, "labels": fake_input.clone()}
        flops, macs, params = get_model_profile(model, kwargs=input_dict, print_profile=True, detailed=True)
        print("FLOPs:", flops)
        print("MACs:", macs)
        print("Params:", params)

        save_number_to_json(
            flops, f"scripts/peftfactory/stat_utils/flops_{path_to_config.split('/')[-1].split('.')[0]}.json"
        )


if __name__ == "__main__":
    fire.Fire(calculate_flops)
