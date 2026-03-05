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

import collections
import re

import numpy as np
from datasets import load_dataset


# preprocess datasets to be loaded by peft-factory


def preprocess_wsc():
    _id2label = {0: "False", 1: "True", -1: ""}

    def _mark_span(text, span_str, span_idx, mark):
        pattern_tmpl = r"^((?:\S+\s){N})(W)"
        pattern = re.sub("N", str(span_idx), pattern_tmpl)
        pattern = re.sub("W", span_str, pattern)
        return re.sub(pattern, rf"\1{mark} \2 {mark}", text)

    def preprocessor(example):
        # converts text as done in T5.
        text = example["text"]
        text = _mark_span(text, example["span1_text"], example["span1_index"], "*")
        # Compensate for 2 added "words" added in previous step.
        span2_index = example["span2_index"] + 2 * int(example["span1_index"] < example["span2_index"])
        input_text = _mark_span(text, example["span2_text"], span2_index, "#")
        label = _id2label[example["label"]]
        return {"inputs": input_text, "targets": label}

    wsc = load_dataset("super_glue", "wsc.fixed")

    return wsc.map(preprocessor)


def preprocess_wic():
    _id2label = {0: "False", 1: "True", -1: ""}

    def preprocessor(example):
        input_text = (
            f"Sentence1: {example['sentence1']}\n\nSentence2: {example['sentence2']}\n\nWord: {example['word']}"
        )
        label = _id2label[example["label"]]
        return {"inputs": input_text, "targets": label}

    wic = load_dataset("super_glue", "wic")

    return wic.map(preprocessor)


def preprocess_multirc():
    _id2label = {0: "False", 1: "True", -1: ""}

    def preprocessor(example):
        input_text = (
            f"Paragraph: {example['paragraph']}\n\nQuestion: {example['question']}\n\nAnswer: {example['answer']}"
        )
        label = _id2label[example["label"]]
        return {"inputs": input_text, "targets": label}

    multirc = load_dataset("super_glue", "multirc")

    return multirc.map(preprocessor)


def preprocess_copa():
    _id2label = {0: "choice1", 1: "choice2", -1: ""}

    def preprocessor(example):
        input_text = f"Premise: {example['premise']}\n\nChoice1: {example['choice1']}\n\nChoice2: {example['choice2']}"
        label = _id2label[example["label"]]
        return {"inputs": input_text, "targets": label}

    copa = load_dataset("super_glue", "copa")

    return copa.map(preprocessor)


def preprocess_record():
    def preprocessor(batch):
        new_batch = collections.defaultdict(list)
        keys = batch.keys()
        for values in zip(*batch.values()):
            ex = dict(zip(keys, values))
            # print(ex["entities"])
            # updates the passage.
            passage = ex["passage"]
            passage = re.sub(r"(\.|\?|\!|\"|\')\n@highlight\n", r"\1 ", passage)
            passage = re.sub(r"\n@highlight\n", ". ", passage)
            inputs = f"Query: {ex['query']}\n\nEntities: {', '.join(ex['entities'])}\n\nPassage: {passage}"

            # duplicates the samples based on  number of answers.
            num_answers = len(ex["answers"])
            num_duplicates = np.maximum(1, num_answers)
            new_batch["inputs"].extend([inputs] * num_duplicates)
            new_batch["targets"].extend(ex["answers"] if num_answers > 0 else ["<unk>"])
            new_batch["idx"].extend([ex["idx"]] * num_duplicates)
            new_batch["answers"].extend([ex["answers"] if num_answers > 0 else ["<unk>"]] * num_duplicates)

        # print(new_batch)
        return new_batch

    record = load_dataset("super_glue", "record")
    return record.map(preprocessor, batched=True, remove_columns=record["train"].column_names)


def preprocess_mmlu():
    _id2label = {0: "A", 1: "B", 2: "C", 3: "D", -1: ""}

    def preprocessor(example):
        input_text = f"Question: {example['question']}\n\nChoices:\nA: {example['choices'][0]}\nB: {example['choices'][1]}\nC: {example['choices'][2]}\nD: {example['choices'][3]}"
        label = _id2label[example["answer"]]

        return {"inputs": input_text, "targets": label}

    mmlu = load_dataset("cais/mmlu", "all")
    return mmlu.map(preprocessor)


def preprocess_piqa():
    _id2label = {0: "solution1", 1: "solution2", -1: ""}

    def preprocessor(example):
        input_text = f"Question: {example['goal']}\n\nSolution1: {example['sol1']}\nSolution2: {example['sol2']}"
        label = _id2label[example["label"]]

        return {"inputs": input_text, "targets": label}

    piqa = load_dataset("ybisk/piqa")
    return piqa.map(preprocessor)


def preprocess_siqa():
    _id2label = {"1": "A", "2": "B", "3": "C", "": ""}

    def preprocessor(example):
        input_text = f"Context: {example['context']}\n\nQuestion: {example['question']}\n\nChoices:\nA: {example['answerA']}\nB: {example['answerB']}\nC: {example['answerC']}"
        label = _id2label[example["label"]]

        return {"inputs": input_text, "targets": label}

    siqa = load_dataset("allenai/social_i_qa")
    return siqa.map(preprocessor)


def preprocess_hellaswag():
    _id2label = {"0": "ending1", "1": "ending2", "2": "ending3", "3": "ending4", "": ""}

    def preprocessor(example):
        input_text = f"Sentence: {example['ctx_a']} {example['ctx_b'].capitalize()},\n\nEnding1: {example['endings'][0]}\nEnding2: {example['endings'][1]}\nEnding3: {example['endings'][2]}\nEnding4: {example['endings'][3]}"
        label = _id2label[example["label"]]

        return {"inputs": input_text, "targets": label}

    hellaswag = load_dataset("Rowan/hellaswag")
    return hellaswag.map(preprocessor)


def preprocess_winogrande():
    _id2label = {"1": "option1", "2": "option2", "": ""}

    def preprocessor(example):
        input_text = f"Sentence: {example['sentence']}\n\nOption1: {example['option1']}\nOption1: {example['option2']}"
        label = _id2label[example["answer"]]

        return {"inputs": input_text, "targets": label}

    winogrande = load_dataset("allenai/winogrande", "winogrande_xl")
    return winogrande.map(preprocessor)


def preprocess_openbookqa():
    def preprocessor(example):
        input_text = f"Question: {example['question_stem']}\n\nChoices:\nA: {example['choices']['text'][0]}\nB: {example['choices']['text'][1]}\nC: {example['choices']['text'][2]}\nD: {example['choices']['text'][3]}"
        label = example["answerKey"]

        return {"inputs": input_text, "targets": label}

    winogrande = load_dataset("allenai/openbookqa", "main")
    return winogrande.map(preprocessor)


def preprocess_math_qa():
    def preprocessor(example):
        input_text = f"Problem: {example['Problem']}\n\nOptions: {example['options']}"
        label = example["correct"]

        return {"inputs": input_text, "targets": label}

    math_qa = load_dataset("allenai/math_qa", "main")
    return math_qa.map(preprocessor)


def preprocess_svamp():
    def preprocessor(example):
        input_text = f"Question: {example['question_concat']}"
        label = f"{example['Equation']} = {example['Answer']}"

        return {"inputs": input_text, "targets": label}

    svamp = load_dataset("ChilleD/SVAMP")
    return svamp.map(preprocessor)


def preprocess_apps():
    def remove_surrogates(text):
        return "".join(c for c in text if not ("\ud800" <= c <= "\udfff"))

    def preprocessor(batch):
        new_batch = collections.defaultdict(list)
        keys = batch.keys()

        for values in zip(*batch.values()):
            ex = dict(zip(keys, values))

            # updates the passage.
            inputs = f"{ex['question']}\n\n{ex['starter_code']}"
            if ex["solutions"] != "":
                ex["solutions"] = eval(ex["solutions"])

            for i, _ in enumerate(ex["solutions"]):
                ex["solutions"][i] = remove_surrogates(ex["solutions"][i])

            inputs = ex["question"]

            num_answers = len(ex["solutions"])
            num_duplicates = np.maximum(1, num_answers)
            new_batch["inputs"].extend([inputs] * num_duplicates)
            new_batch["targets"].extend(ex["solutions"] if num_answers > 0 else [""])

            # new_batch["solutions"].extend([ex["solutions"] if num_answers > 0 else ["<unk>"]] * num_duplicates)

        return new_batch

    apps = load_dataset("codeparrot/apps")
    return apps.map(preprocessor, batched=True, remove_columns=apps["train"].column_names)


# wsc = preprocess_wsc()
# wsc.push_to_hub("rbelanec/wsc")

# wic = preprocess_wic()
# wic.push_to_hub("rbelanec/wic")

# multirc = preprocess_multirc()
# multirc.push_to_hub("rbelanec/multirc")

# copa = preprocess_copa()
# copa.push_to_hub("rbelanec/copa")

# record = preprocess_record()
# record.push_to_hub("rbelanec/record")

# mmlu = preprocess_mmlu()
# mmlu.push_to_hub("rbelanec/mmlu")

# piqa = preprocess_piqa()
# piqa.push_to_hub("rbelanec/piqa")

# siqa = preprocess_siqa()
# siqa.push_to_hub("rbelanec/siqa")

# hellaswag = preprocess_hellaswag()
# hellaswag.push_to_hub("rbelanec/hellaswag")

# winogrande = preprocess_winogrande()
# winogrande.push_to_hub("rbelanec/winogrande")

# openbookqa = preprocess_openbookqa()
# openbookqa.push_to_hub("rbelanec/openbookqa")

# math_qa = preprocess_math_qa()
# math_qa.push_to_hub("rbelanec/math_qa")

# svamp = preprocess_svamp()
# svamp.push_to_hub("rbelanec/svamp")

apps = preprocess_apps()
apps.push_to_hub("rbelanec/apps")
