# Copyright 2025 the LlamaFactory team.
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

from dataclasses import fields
from typing import TYPE_CHECKING

from ...extras.constants import (
    ADAPTERS_CONFIG_MAPPING,
    CUSTOM_PEFT_CONFIG_MAPPING,
    PEFT_CONFIG_MAPPING,
)
from ...extras.packages import is_gradio_available
from ..common import DEFAULT_DATA_DIR
from ..control import list_datasets
from ..locales import LOCALES
from .data import create_preview_box


if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component

    from ..engine import Engine


def create_eval_tab(engine: "Engine") -> dict[str, "Component"]:
    input_elems = engine.manager.get_base_elems()
    elem_dict = dict()

    with gr.Row():
        dataset_dir = gr.Textbox(value=DEFAULT_DATA_DIR, scale=2)
        dataset = gr.Dropdown(multiselect=True, allow_custom_value=True, scale=4)
        preview_elems = create_preview_box(dataset_dir, dataset)

    input_elems.update({dataset_dir, dataset})
    elem_dict.update(dict(dataset_dir=dataset_dir, dataset=dataset, **preview_elems))

    with gr.Row():
        compute_classification_metrics = gr.Checkbox(value=False)
        compute_pscp = gr.Checkbox(value=False)

    input_elems.update({compute_classification_metrics, compute_pscp})
    elem_dict.update(dict(compute_classification_metrics=compute_classification_metrics, compute_pscp=compute_pscp))

    with gr.Accordion(open=False) as pscp_tab:
        elem_dict.update(dict(pscp_tab=pscp_tab))

        with gr.Row():
            pscp_memory = gr.Number(value=5.6)
            pscp_cp = gr.Number(value=5e8)
            pscp_cf = gr.Number(value=10)
            pscp_cm = gr.Number(value=94)
            pscp_bp = gr.Number(value=1.0)
            pscp_bf = gr.Number(value=1.0)
            pscp_bm = gr.Number(value=1.0)

        input_elems.update({pscp_memory, pscp_cp, pscp_cf, pscp_cm, pscp_bp, pscp_bf, pscp_bm})
        elem_dict.update(
            dict(
                pscp_memory=pscp_memory,
                pscp_cp=pscp_cp,
                pscp_cf=pscp_cf,
                pscp_cm=pscp_cm,
                pscp_bp=pscp_bp,
                pscp_bf=pscp_bf,
                pscp_bm=pscp_bm,
            )
        )

    with gr.Accordion(open=False) as peft_tab:
        with gr.Row():
            task_type = gr.Dropdown(
                choices=["SEQ_CLS", "SEQ_2_SEQ_LM", "CAUSAL_LM", "TOKEN_CLS", "QUESTION_ANS", "FEATURE_EXTRACTION"],
                value="CAUSAL_LM",
            )
            inference_mode = gr.Checkbox()

        elem_dict.update(
            dict(
                peft_tab=peft_tab,
                task_type=task_type,
                inference_mode=inference_mode,
            )
        )

        input_elems.update(
            {
                task_type,
                inference_mode,
            }
        )

        peft_common_config_values = [
            "base_model_name_or_path",
            "revision",
            "peft_type",
            "task_type",
            "inference_mode",
            "auto_mapping",
            "num_transformer_submodules",
            "num_attention_heads",
            "num_layers",
            "modules_to_save",
            "token_dim",
        ]
        for peft_config_name in PEFT_CONFIG_MAPPING:
            with gr.Accordion(open=False) as peft_method_tab:
                peft_name = peft_config_name.lower().replace(" ", "_")

                elem_dict.update({peft_name: peft_method_tab})

                LOCALES.update(
                    {
                        peft_name: {
                            "en": {"label": f"{peft_config_name} configurations"},
                            "ru": {"label": f"{peft_config_name} configurations"},
                            "zh": {"label": f"{peft_config_name} configurations"},
                            "ko": {"label": f"{peft_config_name} configurations"},
                            "ja": {"label": f"{peft_config_name} configurations"},
                        }
                    }
                )

                for field in fields(PEFT_CONFIG_MAPPING[peft_config_name]):
                    if field.name in peft_common_config_values:
                        continue

                    with gr.Row():
                        if field.type is bool:
                            elem = gr.Checkbox()
                        if field.type in [float, int]:
                            elem = gr.Number()
                        else:
                            elem = gr.Textbox()

                        elem_dict.update({f"{peft_name}_{field.name}": elem})
                        input_elems.update({elem})

                        LOCALES.update(
                            {
                                f"{peft_name}_{field.name}": {
                                    "en": {"label": field.name},
                                    "ru": {"label": field.name},
                                    "zh": {"label": field.name},
                                    "ko": {"label": field.name},
                                    "ja": {"label": field.name},
                                }
                            }
                        )

        for peft_config_name in ADAPTERS_CONFIG_MAPPING:
            with gr.Accordion(open=False) as peft_method_tab:
                peft_name = peft_config_name.lower().replace(" ", "_")

                elem_dict.update({peft_name: peft_method_tab})

                LOCALES.update(
                    {
                        peft_name: {
                            "en": {"label": f"{peft_config_name} configurations"},
                            "ru": {"label": f"{peft_config_name} configurations"},
                            "zh": {"label": f"{peft_config_name} configurations"},
                            "ko": {"label": f"{peft_config_name} configurations"},
                            "ja": {"label": f"{peft_config_name} configurations"},
                        }
                    }
                )

                for field in fields(ADAPTERS_CONFIG_MAPPING[peft_config_name]):
                    if field.name in peft_common_config_values:
                        continue

                    with gr.Row():
                        if field.type is bool:
                            elem = gr.Checkbox()
                        else:
                            elem = gr.Textbox()

                        elem_dict.update({f"{peft_name}_{field.name}": elem})
                        input_elems.update({elem})

                        LOCALES.update(
                            {
                                f"{peft_name}_{field.name}": {
                                    "en": {"label": field.name},
                                    "ru": {"label": field.name},
                                    "zh": {"label": field.name},
                                    "ko": {"label": field.name},
                                    "ja": {"label": field.name},
                                }
                            }
                        )

        for peft_config_name in CUSTOM_PEFT_CONFIG_MAPPING:
            with gr.Accordion(open=False) as peft_method_tab:
                peft_name = peft_config_name.lower().replace(" ", "_")

                elem_dict.update({peft_name: peft_method_tab})

                LOCALES.update(
                    {
                        peft_name: {
                            "en": {"label": f"{peft_config_name} configurations"},
                            "ru": {"label": f"{peft_config_name} configurations"},
                            "zh": {"label": f"{peft_config_name} configurations"},
                            "ko": {"label": f"{peft_config_name} configurations"},
                            "ja": {"label": f"{peft_config_name} configurations"},
                        }
                    }
                )

                for field in fields(CUSTOM_PEFT_CONFIG_MAPPING[peft_config_name]):
                    if field.name in peft_common_config_values:
                        continue

                    with gr.Row():
                        if field.type is bool:
                            elem = gr.Checkbox()
                        else:
                            elem = gr.Textbox()

                        elem_dict.update({f"{peft_name}_{field.name}": elem})
                        input_elems.update({elem})

                        LOCALES.update(
                            {
                                f"{peft_name}_{field.name}": {
                                    "en": {"label": field.name},
                                    "ru": {"label": field.name},
                                    "zh": {"label": field.name},
                                    "ko": {"label": field.name},
                                    "ja": {"label": field.name},
                                }
                            }
                        )

    with gr.Row():
        cutoff_len = gr.Slider(minimum=4, maximum=131072, value=1024, step=1)
        max_samples = gr.Textbox(value="100000")
        batch_size = gr.Slider(minimum=1, maximum=1024, value=2, step=1)
        predict = gr.Checkbox(value=True)

    input_elems.update({cutoff_len, max_samples, batch_size, predict})
    elem_dict.update(dict(cutoff_len=cutoff_len, max_samples=max_samples, batch_size=batch_size, predict=predict))

    with gr.Row():
        max_new_tokens = gr.Slider(minimum=8, maximum=4096, value=512, step=1)
        top_p = gr.Slider(minimum=0.01, maximum=1, value=0.7, step=0.01)
        temperature = gr.Slider(minimum=0.01, maximum=1.5, value=0.95, step=0.01)
        output_dir = gr.Textbox()

    input_elems.update({max_new_tokens, top_p, temperature, output_dir})
    elem_dict.update(dict(max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature, output_dir=output_dir))

    with gr.Row():
        cmd_preview_btn = gr.Button()
        start_btn = gr.Button(variant="primary")
        stop_btn = gr.Button(variant="stop")

    with gr.Row():
        resume_btn = gr.Checkbox(visible=False, interactive=False)
        progress_bar = gr.Slider(visible=False, interactive=False)

    with gr.Row():
        output_box = gr.Markdown()

    elem_dict.update(
        dict(
            cmd_preview_btn=cmd_preview_btn,
            start_btn=start_btn,
            stop_btn=stop_btn,
            resume_btn=resume_btn,
            progress_bar=progress_bar,
            output_box=output_box,
        )
    )
    output_elems = [output_box, progress_bar]

    cmd_preview_btn.click(engine.runner.preview_eval, input_elems, output_elems, concurrency_limit=None)
    start_btn.click(engine.runner.run_eval, input_elems, output_elems)
    stop_btn.click(engine.runner.set_abort)
    resume_btn.change(engine.runner.monitor, outputs=output_elems, concurrency_limit=None)

    dataset.focus(list_datasets, [dataset_dir], [dataset], queue=False)

    return elem_dict
