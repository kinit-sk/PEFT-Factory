# Copyright 2025 the PEFTFactory team.
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

import glob
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# models = ["gemma-3-1b-it", "llama-3-8b-instruct", "mistral-7b-instruct"]
models = ["llama-3-8b-instruct"]
methods = ["base", "ia3", "lora", "lntuning", "prompt-tuning", "prefix-tuning", "p-tuning"]
# methods = ["prefix-tuning"]
# methods = ["base"]
datasets = [
    "mnli",
    "qqp",
    "qnli",
    "sst2",
    "stsb",
    "mrpc",
    "rte",
    "cola",
    "record",
    "multirc",
    "boolq",
    "wic",
    "wsc",
    "cb",
    "copa",
] + [
    "mmlu",
    "piqa",
    "siqa",
    "hellaswag",
    "winogrande",
    "openbookqa",
    "math_qa",
    "gsm8k",
    "svamp",
    "conala",
    "codealpacapy",
    "apps",
]
# datasets = ["mmlu", "piqa", "siqa", "hellaswag", "winogrande", "openbookqa", "math_qa", "gsm8k", "svamp", "conala", "codealpacapy", "apps"]
# datasets = ["record", "multirc", "boolq", "wic", "wsc", "cb", "copa"]
seeds = [42, 123, 456, 789, 101112]

methods_map = {
    "base": "Base",
    "ia3": "IA3",
    "lora": "LoRA",
    "lntuning": "LNTuning",
    "prompt-tuning": "Prompt Tuning",
    "prefix-tuning": "Prefix Tuning",
    "p-tuning": "P-Tuning",
}

datasets_map = {
    "mnli": "MNLI",
    "qqp": "QQP",
    "qnli": "QNLI",
    "sst2": "SST-2",
    "stsb": "STSB",
    "mrpc": "MRPC",
    "rte": "RTE",
    "cola": "CoLA",
    "record": "ReCoRD",
    "multirc": "MultiRC",
    "boolq": "BoolQ",
    "wic": "WiC",
    "wsc": "WSC",
    "cb": "CB",
    "copa": "COPA",
    "mmlu": "MMLU",
    "piqa": "PIQA",
    "siqa": "SIQA",
    "hellaswag": "HellaSwag",
    "winogrande": "Winogrande",
    "openbookqa": "OpenBookQA",
    "math_qa": "MathQA",
    "gsm8k": "GSM8K",
    "svamp": "SVAMP",
    "conala": "CoNaLa",
    "codealpacapy": "CodeAlpacaPy",
    "apps": "APPS",
}


def plot_barplot(df, title="Performance Comparison", basename="barplot"):
    """Create and save a barplot from a dataframe where each cell is a dict with 'mean' and 'std' keys.

    Args:
        df (pd.DataFrame): DataFrame with dict values {'mean': x, 'std': y}
        title (str): Title of the plot
        basename (str): Base name (without extension) for output files
    """
    means = df.map(lambda x: x["mean"])
    stds = df.map(lambda x: x["std"])

    # Plot
    ax = means.plot(kind="bar", yerr=stds, capsize=4, figsize=(12, 4), rot=0)

    ax.set_xticklabels(ax.get_xticklabels())

    # Titles and labels
    # plt.title(title, fontsize=14)
    plt.ylabel("F1 (%)", fontsize=12)
    # plt.xlabel("Method", fontsize=12, fontweight="bold")

    # Legend under the image, single row
    plt.legend(
        # title="Dataset",
        bbox_to_anchor=(0.5, -0.1),  # move under plot
        loc="upper center",
        ncol=len(df.columns),  # all in one line
        frameon=False,
    )

    plt.tight_layout()

    # Save in both formats
    plt.savefig(f"{basename}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{basename}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved barplot as {basename}.png and {basename}.pdf")


def get_single_result(results, dataset):
    print(dataset)
    if "macro_f1" in results:
        return results["macro_f1"]
    elif "pearsonr" in results:
        return results["pearsonr"]
    elif dataset in ["gsm8k", "svamp"]:
        return results["accuracy"]
    elif dataset in ["conala", "codealpacapy", "apps"]:
        return results["codebleu"]
    else:
        return results["f1"]


def get_results_from_jsonl(eval_dir):
    results = {}
    with open(f"{eval_dir}/results.jsonl") as json_file:
        for line in json_file:
            results.update(json.loads(line))

    return results


def _format_mean_std(cell):
    """Format mean/std dicts into a LaTeX-friendly string."""
    if not isinstance(cell, dict):
        return ""

    mean = cell.get("mean")
    std = cell.get("std")

    if mean is None:
        return ""

    if std is None or np.isnan(std):
        return f"${mean:.1f}$"

    return f"${mean:.1f} \\pm {std:.1f}$"


for m in models:
    print(f"Model {m}")

    results = {}
    for pm in methods:
        print(f"Method {pm}")
        results[pm] = {}
        for d in datasets:
            print(f"Dataset {d}")
            glob_res = glob.glob(f"saves/{pm}/{m}/eval_{d}*")

            if not glob_res:
                continue

            try:
                results[pm][d] = get_single_result(get_results_from_jsonl(sorted(glob_res)[-1]), d) * 100
            except FileNotFoundError:
                continue

    results_df = pd.DataFrame(results).T
    print(
        results_df.to_latex(
            float_format="%.1f", caption="Performance across tasks and tuning methods", label="tab:results"
        )
    )

    print(results_df.T.mean().round(1))


stability_datasets = ["cb", "copa", "svamp", "cola", "sst2", "hellaswag", "wsc"]

for m in models:
    print(f"Model {m}")

    results = {}
    for pm in ["ia3", "lora", "lntuning", "prompt-tuning", "prefix-tuning", "p-tuning"]:
        print(f"Method {pm}")
        results[methods_map[pm]] = {}
        for d in stability_datasets:
            print(f"Dataset {d}")
            results[methods_map[pm]][datasets_map[d]] = {}
            seed_results = []
            for s in seeds:
                glob_res = glob.glob(f"saves_multiple/{pm}/{m}/eval_{d}_{s}*")

                if not glob_res:
                    continue

                try:
                    seed_results.append(get_single_result(get_results_from_jsonl(sorted(glob_res)[-1]), d) * 100)
                except FileNotFoundError:
                    continue

            if seed_results:
                results[methods_map[pm]][datasets_map[d]]["mean"] = np.mean(seed_results)
                results[methods_map[pm]][datasets_map[d]]["std"] = np.std(seed_results)

    results_df = pd.DataFrame(results).T
    # print(results_df.to_string())
    formatted_df = results_df.applymap(_format_mean_std)
    print(
        formatted_df.to_latex(caption="Performance across tasks and tuning methods", label="tab:results", escape=False)
    )
    plot_barplot(results_df, title="")


### PEFT methods for PEFT-Factory demonstration
models = ["llama-3.2-1b-instruct"]
methods = ["ia3", "bitfit", "prefix-tuning"]
datasets = ["sst2", "cola", "wsc", "svamp"]

for m in models:
    print(f"Model {m}")

    results = {}
    for pm in methods:
        print(f"Method {pm}")
        results[pm] = {}
        for d in datasets:
            print(f"Dataset {d}")
            glob_res = glob.glob(f"saves_multiple/{pm}/{m}/eval_{d}*")

            if not glob_res:
                continue

            try:
                results[pm][d] = get_single_result(get_results_from_jsonl(sorted(glob_res)[-1]), d) * 100
            except FileNotFoundError:
                continue

    results_df = pd.DataFrame(results).T
    print(
        results_df.to_latex(
            float_format="%.1f", caption="Performance across tasks and tuning methods", label="tab:results"
        )
    )

    print(results_df.T.mean().round(1))
