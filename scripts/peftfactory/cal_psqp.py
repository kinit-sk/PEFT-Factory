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


llama_base_flops = 8.38

methods = ["ia3", "prompt-tuning", "prefix-tuning", "p-tuning", "lora", "lntuning"]
methods_parameters_map = {
    "ia3": 196608,
    "prompt-tuning": 409600,
    "prefix-tuning": 34177536,
    "p-tuning": 53130752,
    "lora": 14680064,
    "lntuning": 266240,
}

methods_flops_map = {
    "ia3": 8.69 - llama_base_flops,
    "prompt-tuning": 10.05 - llama_base_flops,
    "prefix-tuning": 8.39 - llama_base_flops,
    "p-tuning": 10.06 - llama_base_flops,
    "lora": 8.38 - llama_base_flops,
    "lntuning": 8.38 - llama_base_flops,
}

methods_memory_map = {
    "ia3": 28.41,
    "prompt-tuning": 42.47,
    "prefix-tuning": 27.61,
    "p-tuning": 34.53,
    "lora": 27.75,
    "lntuning": 27.43,
}

performance_map = {
    "ia3": 74.7,
    "prompt-tuning": 50,
    "prefix-tuning": 45.9,
    "p-tuning": 51.7,
    "lora": 80.1,
    "lntuning": 77.8,
}


def cost(m, c, beta):
    return (1 + m / c) ** -beta


def psqp(p, cost_params, cost_flops, cost_memory):
    return p * cost_params * cost_flops * cost_memory


def highlight(val, best, second):
    if val == best:
        return f"\\textbf{{{val}}}"
    elif val == second:
        return f"\\underline{{{val}}}"
    else:
        return str(val)


def print_latex_table(df: pd.DataFrame, beta_p, beta_f, beta_m):
    df = df.copy()

    name_map = {
        "ia3": "IA3",
        "prompt-tuning": "Prompt Tuning",
        "prefix-tuning": "Prefix Tuning",
        "p-tuning": "P-Tuning",
        "lora": "LoRA",
        "lntuning": "LNTuning",
    }

    df["latex_name"] = df["method"].map(name_map)

    best, second = (
        psqp_df["psqp"].sort_values(ascending=False).iloc[0],
        psqp_df["psqp"].sort_values(ascending=False).iloc[1],
    )

    df["psqp_fmt"] = df["psqp"].apply(lambda x: highlight(x, best, second))

    latex_lines = []
    latex_lines.append("\\begin{table}[ht]")
    latex_lines.append("\\centering")
    latex_lines.append("\\resizebox{\\columnwidth}{!}{%")
    latex_lines.append("\\begin{tabular}{@{}l|lllll@{}}")
    latex_lines.append("\\toprule")
    latex_lines.append(
        "PEFT Method & $P_{avg}$ & $cost_p^{-"
        + str(beta_p)
        + "}$ & $cost_f^{-"
        + str(beta_f)
        + "}$ & $cost_m^{-"
        + str(beta_m)
        + "}$ & PSCP \\\\ \\midrule"
    )

    for _, row in df.iterrows():
        latex_lines.append(
            f"{row['latex_name']} & {row['performance']} & {row['params']} & {row['flops']} & {row['memory']} & {row['psqp_fmt']} \\\\"
        )

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}%")
    latex_lines.append("}")
    latex_lines.append("\\caption{}")
    latex_lines.append("\\label{}")
    latex_lines.append("\\end{table}")

    latex_table = "\n".join(latex_lines)
    print(latex_table)


# c_p = np.median(list(methods_parameters_map.values()))
# c_p = np.exp(np.mean(np.log(list(methods_parameters_map.values()))))
# c_p = np.max(list(methods_parameters_map.values()))
c_p = 5e8  #
c_f = 10  # such slowdown would mean that its better to use the base model
c_m = 94  # h100 nvl has 94GB


def cal_psqps(c_p, c_f, c_m, beta_p, beta_f, beta_m):
    rows = []
    for method in methods:
        cost_params = cost(methods_parameters_map[method], c_p, beta_p)
        cost_flops = cost(methods_flops_map[method], c_f, beta_f)
        cost_memory = cost(methods_memory_map[method], c_m, beta_m)

        psqp_val = np.round(psqp(performance_map[method], cost_params, cost_flops, cost_memory), 2)

        rows.append(
            {
                "method": method,
                "performance": performance_map[method],
                "params": np.round(cost_params, 2),
                "flops": np.round(cost_flops, 2),
                "memory": np.round(cost_memory, 2),
                "psqp": psqp_val,
            }
        )

    return pd.DataFrame(rows)


beta_p = 1
beta_f = 1
beta_m = 1

psqp_df = cal_psqps(c_p, c_f, c_m, beta_p, beta_f, beta_m)


print(psqp_df)

print_latex_table(psqp_df, beta_p, beta_f, beta_m)

betas = [(0.5, 1, 1), (1, 0.5, 1), (1, 1, 0.5), (2, 1, 1), (1, 2, 1), (1, 1, 2)]
# betas = [(1,1,1), (2,2,2), (3,3,3), (4,4,4)]
for beta_p, beta_f, beta_m in betas:
    psqp_df = cal_psqps(c_p, c_f, c_m, beta_p, beta_f, beta_m)
    print(f"Beta params: {beta_p}, Beta flops: {beta_f}, Beta memory: {beta_m}")
    print_latex_table(psqp_df, beta_p, beta_f, beta_m)
    print("\n\n")
