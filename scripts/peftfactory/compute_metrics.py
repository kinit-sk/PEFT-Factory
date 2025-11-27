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

import argparse
import ast
import json
import math
import operator
import re

import evaluate
import numpy as np
from codebleu import calc_codebleu
from datasets import load_dataset
from sklearn.metrics import f1_score


def string_to_float(string, default=-1.0):
    """Converts string to float, using default when conversion not possible."""
    try:
        return float(string)
    except ValueError:
        return default


def check_data_state(preds, targets):
    assert len(preds) == len(targets)


def binary_reverse(targets, labels):
    return [labels[0] if target == labels[1] else labels[1] for target in targets]


def em(preds, targets, labels):
    check_data_state(preds, targets)

    preds, targets = np.asarray(preds, dtype="<U16"), np.asarray(targets, dtype="<U16")

    return {"exact_match": np.sum(preds == targets) / preds.size}


# def f1(preds, targets, labels):
#     check_data_state(preds, targets)

#     preds, targets = np.asarray(preds, dtype="<U16"), np.asarray(targets, dtype="<U16")

#     invalid_idx_mask = np.logical_and(preds != labels[0], preds != labels[1])

#     preds[invalid_idx_mask] = binary_reverse(targets[invalid_idx_mask], labels)

#     return {"f1": f1_score(targets, preds, labels=labels, pos_label=labels[0])}


# def macro_f1(preds, targets, labels):
#     check_data_state(preds, targets)

#     preds, targets = np.asarray(preds, dtype="<U16"), np.asarray(targets, dtype="<U16")

#     invalid_idx_mask = np.logical_and(preds != labels[0], preds != labels[1])

#     preds[invalid_idx_mask] = binary_reverse(targets[invalid_idx_mask], labels)

#     return {"macro_f1": f1_score(targets, preds, labels=labels, average="macro")}


def f1(preds, targets, labels):
    preds, targets = np.asarray(preds, dtype="<U16"), np.asarray(targets, dtype="<U16")

    labels = [x.lower() for x in labels]

    invalid_idx_mask = np.logical_and(preds != labels[0], preds != labels[1])

    preds[invalid_idx_mask] = binary_reverse(targets[invalid_idx_mask], labels)

    label_to_id = {label: idx for idx, label in enumerate(labels)}

    # print(label_to_id)

    targets = list(map(label_to_id.get, targets))
    preds = list(map(label_to_id.get, preds))

    # print(preds, targets)

    return {"f1": f1_score(targets, preds)}


def macro_f1(preds, targets, labels):
    preds, targets = np.asarray(preds, dtype="<U16"), np.asarray(targets, dtype="<U16")

    labels = [x.lower() for x in labels]

    invalid_idx_mask = ~np.isin(preds, labels)

    preds[invalid_idx_mask] = binary_reverse(targets[invalid_idx_mask], labels)

    label_to_id = {label: idx for idx, label in enumerate(labels)}

    # print(label_to_id)

    targets = list(map(label_to_id.get, targets))
    preds = list(map(label_to_id.get, preds))

    # print(preds, targets)

    return {"macro_f1": f1_score(targets, preds, average="macro")}


def pearsonr(preds, targets, labels):
    metric = evaluate.load("pearsonr")

    targets = [string_to_float(t) for t in targets]
    preds = [string_to_float(p) for p in preds]

    return metric.compute(predictions=preds, references=targets)


def spearmanr(preds, targets, labels):
    metric = evaluate.load("spearmanr")

    targets = [string_to_float(t) for t in targets]
    preds = [string_to_float(p) for p in preds]

    return metric.compute(predictions=preds, references=targets)


def record(preds):
    dataset = load_dataset("rbelanec/record", split="validation")
    metric = evaluate.load("super_glue", "record")

    predictions = [{"idx": dataset[i]["idx"], "prediction_text": p} for i, p in enumerate(preds)]

    references = [{"idx": d["idx"], "answers": d["answers"]} for d in dataset]

    return metric.compute(predictions=predictions, references=references)


def gsm8k(preds, targets, labels):
    def extract_final_answer(text):
        if "####" in text:
            return text.split("####")[-1].strip()
        return None

    def normalize_answer(ans):
        try:
            return float(ans.replace(",", "").replace("%", ""))
        except (AttributeError, ValueError):
            return None

    format_errors = 0
    correct = 0
    total = 0
    for i in range(len(preds)):
        pred = normalize_answer(extract_final_answer(preds[i]))
        gold = normalize_answer(extract_final_answer(targets[i]))
        if pred is not None and gold is not None:
            if abs(pred - gold) < 1e-6:  # allow tiny float error
                correct += 1

        if pred is None or gold is None:
            format_errors += 1

        total += 1

    accuracy = correct / total
    return {"accuracy": accuracy, "format_errors": format_errors}


def svamp(preds, targets, labels):
    OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def safe_eval(expr):
        def _eval(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value

            if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
                return OPS[type(node.op)](_eval(node.operand))

            if isinstance(node, ast.BinOp) and type(node.op) in OPS:
                return OPS[type(node.op)](_eval(node.left), _eval(node.right))

            if isinstance(node, ast.Expr):
                return _eval(node.value)

            raise ValueError("Disallowed expression")

        tree = ast.parse(expr.strip(), mode="eval")
        return float(_eval(tree.body))

    def is_single_outer_parens(s):
        s = s.strip()
        if not (s.startswith("(") and s.endswith(")")):
            return False

        depth = 0
        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    return False  # outer pair closes before end -> not a single outer wrapper
        return depth == 0

    num_re = re.compile(r"^[+-]?(\d+(\.\d+)?|\.\d+)$")

    def parse_equation(eq: str):
        if eq.count("=") != 1:
            return None, None, "not_single_equation"
        lhs, rhs = [p.strip() for p in eq.split("=", 1)]
        if not is_single_outer_parens(lhs):
            return None, None, "lhs_not_single_parenthesized"
        if not num_re.match(rhs.replace(",", "")):
            return None, None, "rhs_not_numeric"
        return lhs, rhs, None

    def normalize_number_str(s):
        x = float(s.replace(",", ""))
        return str(int(x)) if math.isclose(x, round(x), rel_tol=0, abs_tol=1e-9) else str(x)

    def canon(eq):
        # remove spaces and normalize RHS number to canonical form
        lhs, rhs = [p.strip() for p in eq.split("=", 1)]
        sub = re.sub(r"\\s+", "", lhs)
        return f"{sub}={normalize_number_str(rhs)}"

    def evaluate_svamp(pred, gold=None, tol=1e-6):
        lhs, rhs, err = parse_equation(pred)
        if err:
            return {"ok": False, "reason": err, "match_gold": False}
        try:
            lhs_val = safe_eval(lhs[1:-1])  # strip outer parentheses
            rhs_val = float(rhs.replace(",", ""))
        except Exception:
            # print(e)
            return {"ok": False, "reason": "eval_error", "match_gold": False}
        math_ok = abs(lhs_val - rhs_val) < tol
        gold_match = (canon(pred) == canon(gold)) if gold else None
        return {"ok": math_ok, "reason": None if math_ok else "math_mismatch", "match_gold": gold_match}

    total = len(preds)
    fmt_errors = 0
    math_correct = 0
    gold_exact = 0

    for i in range(total):
        r = evaluate_svamp(preds[i], targets[i])
        if r["reason"] in {"not_single_equation", "lhs_not_single_parenthesized", "rhs_not_numeric", "eval_error"}:
            fmt_errors += 1
            # print(r)
            # print(preds[i])
        if r["ok"]:
            math_correct += 1
        if r["ok"] and r["match_gold"]:
            gold_exact += 1

    accuracy = math_correct / total if total else 0.0

    return {
        "total": total,
        "math_correct": math_correct,
        "format_errors": fmt_errors,
        "gold_exact_equation": gold_exact,
        "accuracy": accuracy,
    }


def codebleu_metric(preds, targets, labels):
    return calc_codebleu(targets, preds, lang="python", weights=(0.25, 0.25, 0.25, 0.25))


DATASET_TO_METRIC_MAPPING = {
    "mnli": {"metrics": [macro_f1, em], "labels": ["entailment", "neutral", "contradiction"]},
    "qqp": {"metrics": [f1, em], "labels": ["not_duplicate", "duplicate"]},
    "qnli": {"metrics": [f1, em], "labels": ["entailment", "not_entailment"]},
    "sst2": {"metrics": [f1, em], "labels": ["negative", "positive"]},
    "stsb": {"metrics": [pearsonr, spearmanr], "labels": []},
    "mrpc": {"metrics": [f1, em], "labels": ["not_equivalent", "equivalent"]},
    "rte": {"metrics": [f1, em], "labels": ["entailment", "not_entailment"]},
    "cola": {"metrics": [f1, em], "labels": ["unacceptable", "acceptable"]},
    "record": {"metrics": [record], "labels": []},
    "multirc": {"metrics": [f1, em], "labels": ["false", "true"]},
    "boolq": {"metrics": [f1, em], "labels": ["false", "true"]},
    "wic": {"metrics": [f1, em], "labels": ["false", "true"]},
    "wsc": {"metrics": [f1, em], "labels": ["false", "true"]},
    "cb": {"metrics": [macro_f1, em], "labels": ["entailment", "contradiction", "neutral"]},
    "copa": {"metrics": [f1, em], "labels": ["choice1", "choice2"]},
    "mmlu": {"metrics": [macro_f1, em], "labels": ["A", "B", "C", "D"]},
    "piqa": {"metrics": [f1, em], "labels": ["solution1", "solution2"]},
    "siqa": {"metrics": [macro_f1, em], "labels": ["A", "B", "C"]},
    "hellaswag": {"metrics": [macro_f1, em], "labels": ["ending1", "ending2", "ending3", "ending4"]},
    "winogrande": {"metrics": [f1, em], "labels": ["option1", "option2"]},
    "openbookqa": {"metrics": [macro_f1, em], "labels": ["A", "B", "C", "D"]},
    "math_qa": {"metrics": [macro_f1, em], "labels": ["a", "b", "c", "d", "e"]},
    "gsm8k": {"metrics": [gsm8k], "labels": []},
    "svamp": {"metrics": [svamp], "labels": []},
    "conala": {"metrics": [codebleu_metric], "labels": []},
    "codealpacapy": {"metrics": [codebleu_metric], "labels": []},
    "apps": {"metrics": [codebleu_metric], "labels": []},
}

argparse_parser = argparse.ArgumentParser(
    prog="Compute metrics.",
    description="Compute metrics for single model.",
)

argparse_parser.add_argument("eval_dir", help="Directory created during evaluation.")
argparse_parser.add_argument("dataset", help="Dataset used during evaluation.")

args = argparse_parser.parse_args()

eval_dir = args.eval_dir
dataset = args.dataset

eval_samples = []
with open(f"{eval_dir}/generated_predictions.jsonl") as json_file:
    for line in json_file:
        eval_samples.append(json.loads(line))

labels, predictions = [], []
for es in eval_samples:
    labels.append(es["label"].split("</think>\n\n")[-1].strip().lower())
    predictions.append(es["predict"].split("</think>\n\n")[-1].strip().lower())

# print(list(zip(predictions,labels)))

with open(f"{eval_dir}/results.jsonl", "w") as outfile:
    for metric in DATASET_TO_METRIC_MAPPING[dataset]["metrics"]:
        if dataset in ["record"]:
            result = metric(predictions)
        else:
            result = metric(predictions, labels, DATASET_TO_METRIC_MAPPING[dataset]["labels"])

        print(result)
        json.dump(result, outfile)
        outfile.write("\n")
