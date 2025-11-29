# Copyright 2025 HuggingFace Inc., THUDM, and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library and the THUDM's ChatGLM implementation.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
# https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
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
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from deepspeed.accelerator import get_accelerator  # type: ignore
from deepspeed.profiling.flops_profiler import get_model_profile  # type: ignore
from sklearn.metrics import f1_score
from transformers.utils import is_nltk_available

from ...extras.constants import IGNORE_INDEX
from ...extras.misc import numpify
from ...extras.packages import is_jieba_available, is_rouge_available


def _pscp_cost(metric_value: float, constant: float, beta: float) -> float:
    """Compute the scaled cost contribution used by the PSCP metric."""
    return (1.0 + float(metric_value) / float(constant)) ** (-float(beta))


if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer


if is_jieba_available():
    import jieba  # type: ignore


if is_nltk_available():
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # type: ignore


if is_rouge_available():
    from rouge_chinese import Rouge  # type: ignore


def check_data_state(preds, targets):
    assert len(preds) == len(targets)


def eval_logit_processor(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    r"""Compute the token with the largest likelihood to reduce memory footprint."""
    if isinstance(logits, (list, tuple)):
        if logits[0].dim() == 3:  # (batch_size, seq_len, vocab_size)
            logits = logits[0]
        else:  # moe models have aux loss
            logits = logits[1]

    if logits.dim() != 3:
        raise ValueError("Cannot process the logits.")

    return torch.argmax(logits, dim=-1)


def accuracy(preds, targets):
    check_data_state(preds, targets)

    preds, targets = np.asarray(preds, dtype="<U16"), np.asarray(targets, dtype="<U16")

    return np.sum(preds == targets) / preds.size


def calculate_flops(
    model,
):
    r"""Calculate the flops of pre-trained models."""
    batch_size: int = 1
    seq_length: int = 512

    with get_accelerator().device(0):
        fake_input = torch.ones((batch_size, seq_length), dtype=torch.long, device=model.device)
        input_dict = {"input_ids": fake_input, "labels": fake_input.clone()}
        flops, _, _ = get_model_profile(model, kwargs=input_dict, print_profile=False, detailed=False)
        return float(flops.replace(" T", ""))


def pscp(
    performance: float,
    params: float,
    flops: float,
    memory: float,
    c_p: float,
    c_f: float,
    c_m: float,
    beta_p: float = 1.0,
    beta_f: float = 1.0,
    beta_m: float = 1.0,
) -> dict[str, float]:
    """Compute the Parameter-Speed-Cost-Performance (PSCP) metric."""
    cost_params = _pscp_cost(params, c_p, beta_p)
    cost_flops = _pscp_cost(flops, c_f, beta_f)
    cost_memory = _pscp_cost(memory, c_m, beta_m)

    value = performance * cost_params * cost_flops * cost_memory
    return float(np.round(value, 2))


def f1(preds, targets, valid_labels: Optional[list[str]] = None):
    check_data_state(preds, targets)

    if len(preds) == 0:
        return {"f1": 0.0}

    label_pool: list[str]
    if valid_labels is not None:
        # Preserve user-specified label order while removing duplicates.
        label_pool = list(dict.fromkeys(valid_labels))
    else:
        label_pool = sorted(set(targets))

    processed_preds: list[str] = []
    processed_targets: list[str] = []
    unknown_label = "__unknown__"
    unknown_used = False

    for pred_label, target_label in zip(preds, targets):
        processed_targets.append(target_label)
        if pred_label in label_pool:
            processed_preds.append(pred_label)
        else:
            processed_preds.append(unknown_label)
            unknown_used = True

    if unknown_used and unknown_label not in label_pool:
        label_pool.append(unknown_label)

    mapping = {label: idx for idx, label in enumerate(label_pool)}
    y_pred = [mapping[label] for label in processed_preds]
    y_true = [mapping[label] for label in processed_targets]

    score = f1_score(y_true, y_pred, average="macro")
    return score


@dataclass
class ComputeAccuracy:
    r"""Compute accuracy and support `batch_eval_metrics`."""

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        for i in range(len(preds)):
            pred, label = preds[i, :-1], labels[i, 1:]
            label_mask = label != IGNORE_INDEX
            self.score_dict["accuracy"].append(np.mean(pred[label_mask] == label[label_mask]))

        if compute_result:
            return self._dump()


@dataclass
class ComputeSimilarity:
    r"""Compute text similarity scores and support `batch_eval_metrics`.

    Wraps the tokenizer into metric functions, used in CustomSeq2SeqTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                self.score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            self.score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        if compute_result:
            return self._dump()


@dataclass
class ComputeClassification:
    r"""Compute text similarity scores and support `batch_eval_metrics`.

    Wraps the tokenizer into metric functions, used in CustomSeq2SeqTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": [], "f1": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        self.score_dict["accuracy"].append(accuracy(decoded_preds, decoded_labels))
        self.score_dict["f1"].append(f1(decoded_preds, decoded_labels))

        if compute_result:
            return self._dump()
