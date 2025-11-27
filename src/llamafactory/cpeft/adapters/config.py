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

from dataclasses import dataclass

from adapters import DoubleSeqBnConfig, ParBnConfig, SeqBnConfig


@dataclass
class AdaptersConfig:
    adapter_name: str = "default"


@dataclass
class AdaptersDoubleSeqBnConfig(AdaptersConfig, DoubleSeqBnConfig):
    # this is mostly because DoubleSeqBnConfig from adapters contains Union[X, Y] where both are not optional, so HFArgumentParser cannot parse it properly
    reduction_factor: float = 16
    residual_before_ln: bool = True


@dataclass
class AdaptersSeqBnConfig(AdaptersConfig, SeqBnConfig):
    # this is mostly because DoubleSeqBnConfig from adapters contains Union[X, Y] where both are not optional, so HFArgumentParser cannot parse it properly
    reduction_factor: float = 16
    residual_before_ln: bool = True


@dataclass
class AdaptersParBnConfig(AdaptersConfig, ParBnConfig):
    # this is mostly because DoubleSeqBnConfig from adapters contains Union[X, Y] where both are not optional, so HFArgumentParser cannot parse it properly
    reduction_factor: float = 16
    residual_before_ln: bool = True
