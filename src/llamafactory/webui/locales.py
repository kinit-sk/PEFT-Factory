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

LOCALES = {
    "title": {
        "en": {
            "value": "<h1><center>🦙🏭⚙️PEFT-Factory: Unified Parameter-Efficient Fine-Tuning of 100+ LLMs</center></h1>",
        },
        "ru": {
            "value": "<h1><center>🦙🏭⚙️PEFT-Factory: Унифицированная эффективная тонкая настройка 100+ LLMs</center></h1>",
        },
        "zh": {
            "value": "<h1><center>🦙🏭⚙️PEFT-Factory: 一站式大模型高效微调平台</center></h1>",
        },
        "ko": {
            "value": "<h1><center>🦙🏭⚙️PEFT-Factory: 100+ LLMs를 위한 통합 효율적인 튜닝</center></h1>",
        },
        "ja": {
            "value": "<h1><center>🦙🏭⚙️PEFT-Factory: 100+ LLMs の統合効率的なチューニング</center></h1>",
        },
    },
    "subtitle": {
        "en": {
            "value": (
                "<h3><center>Visit <a href='https://github.com/kinit-sk/PEFT-Factory' target='_blank'>"
                "GitHub Page</a></center></h3>"
            ),
        },
        "ru": {
            "value": (
                "<h3><center>Посетить <a href='https://github.com/kinit-sk/PEFT-Factory' target='_blank'>"
                "страницу GitHub</a> <a href='https://peft-factory.readthedocs.io/en/latest/' target='_blank'>"
                "Документацию</a></center></h3>"
            ),
        },
        "zh": {
            "value": (
                "<h3><center>访问 <a href='https://github.com/kinit-sk/PEFT-Factory' target='_blank'>"
                "GitHub 主页</a> <a href='https://peft-factory.readthedocs.io/zh-cn/latest/' target='_blank'>"
                "官方文档</a></center></h3>"
            ),
        },
        "ko": {
            "value": (
                "<h3><center><a href='https://github.com/kinit-sk/PEFT-Factory' target='_blank'>"
                "GitHub 페이지</a> <a href='https://peft-factory.readthedocs.io/en/latest/' target='_blank'>"
                "공식 문서</a>를 방문하세요.</center></h3>"
            ),
        },
        "ja": {
            "value": (
                "<h3><center><a href='https://github.com/kinit-sk/PEFT-Factory' target='_blank'>"
                "GitHub ページ</a> <a href='https://peft-factory.readthedocs.io/en/latest/' target='_blank'>"
                "ドキュメント</a>にアクセスする</center></h3>"
            ),
        },
    },
    "lang": {
        "en": {
            "label": "Language",
        },
        "ru": {
            "label": "Язык",
        },
        "zh": {
            "label": "语言",
        },
        "ko": {
            "label": "언어",
        },
        "ja": {
            "label": "言語",
        },
    },
    "model_name": {
        "en": {
            "label": "Model name",
            "info": "Input the initial name to search for the model.",
        },
        "ru": {
            "label": "Название модели",
            "info": "Введите начальное имя для поиска модели.",
        },
        "zh": {
            "label": "模型名称",
            "info": "输入首单词以检索模型。",
        },
        "ko": {
            "label": "모델 이름",
            "info": "모델을 검색할 초기 이름을 입력하세요.",
        },
        "ja": {
            "label": "モデル名",
            "info": "モデルを検索するための初期名を入力してください。",
        },
    },
    "model_path": {
        "en": {
            "label": "Model path",
            "info": "Path to pretrained model or model identifier from Hugging Face.",
        },
        "ru": {
            "label": "Путь к модели",
            "info": "Путь к предварительно обученной модели или идентификатор модели от Hugging Face.",
        },
        "zh": {
            "label": "模型路径",
            "info": "本地模型的文件路径或 Hugging Face 的模型标识符。",
        },
        "ko": {
            "label": "모델 경로",
            "info": "사전 훈련된 모델의 경로 또는 Hugging Face의 모델 식별자.",
        },
        "ja": {
            "label": "モデルパス",
            "info": "事前学習済みモデルへのパス、または Hugging Face のモデル識別子。",
        },
    },
    "hub_name": {
        "en": {
            "label": "Hub name",
            "info": "Choose the model download source.",
        },
        "ru": {
            "label": "Имя хаба",
            "info": "Выберите источник загрузки модели.",
        },
        "zh": {
            "label": "模型下载源",
            "info": "选择模型下载源。（网络受限环境推荐使用 ModelScope）",
        },
        "ko": {
            "label": "모델 다운로드 소스",
            "info": "모델 다운로드 소스를 선택하세요.",
        },
        "ja": {
            "label": "モデルダウンロードソース",
            "info": "モデルをダウンロードするためのソースを選択してください。",
        },
    },
    "finetuning_type": {
        "en": {
            "label": "Finetuning method",
        },
        "ru": {
            "label": "Метод дообучения",
        },
        "zh": {
            "label": "微调方法",
        },
        "ko": {
            "label": "파인튜닝 방법",
        },
        "ja": {
            "label": "ファインチューニング方法",
        },
    },
    "checkpoint_path": {
        "en": {
            "label": "Checkpoint path",
        },
        "ru": {
            "label": "Путь контрольной точки",
        },
        "zh": {
            "label": "检查点路径",
        },
        "ko": {
            "label": "체크포인트 경로",
        },
        "ja": {
            "label": "チェックポイントパス",
        },
    },
    "quantization_bit": {
        "en": {
            "label": "Quantization bit",
            "info": "Enable quantization (QLoRA).",
        },
        "ru": {
            "label": "Уровень квантования",
            "info": "Включить квантование (QLoRA).",
        },
        "zh": {
            "label": "量化等级",
            "info": "启用量化（QLoRA）。",
        },
        "ko": {
            "label": "양자화 비트",
            "info": "양자화 활성화 (QLoRA).",
        },
        "ja": {
            "label": "量子化ビット",
            "info": "量子化を有効にする (QLoRA)。",
        },
    },
    "quantization_method": {
        "en": {
            "label": "Quantization method",
            "info": "Quantization algorithm to use.",
        },
        "ru": {
            "label": "Метод квантования",
            "info": "Алгоритм квантования, который следует использовать.",
        },
        "zh": {
            "label": "量化方法",
            "info": "使用的量化算法。",
        },
        "ko": {
            "label": "양자화 방법",
            "info": "사용할 양자화 알고리즘.",
        },
        "ja": {
            "label": "量子化方法",
            "info": "使用する量子化アルゴリズム。",
        },
    },
    "template": {
        "en": {
            "label": "Chat template",
            "info": "The chat template used in constructing prompts.",
        },
        "ru": {
            "label": "Шаблон чата",
            "info": "Шаблон чата используемый для составления подсказок.",
        },
        "zh": {
            "label": "对话模板",
            "info": "构建提示词时使用的模板。",
        },
        "ko": {
            "label": "채팅 템플릿",
            "info": "프롬프트 작성에 사용되는 채팅 템플릿.",
        },
        "ja": {
            "label": "チャットテンプレート",
            "info": "プロンプトの構築に使用されるチャットテンプレート。",
        },
    },
    "rope_scaling": {
        "en": {
            "label": "RoPE scaling",
            "info": "RoPE scaling method to use.",
        },
        "ru": {
            "label": "Масштабирование RoPE",
            "info": "Метод масштабирования RoPE для использования.",
        },
        "zh": {"label": "RoPE 插值方法", "info": "RoPE 插值时使用的方法。"},
        "ko": {
            "label": "RoPE 스케일링",
            "info": "사용할 RoPE 스케일링 방법.",
        },
        "ja": {
            "label": "RoPE スケーリング",
            "info": "使用する RoPE スケーリング方法。",
        },
    },
    "booster": {
        "en": {
            "label": "Booster",
            "info": "Approach used to boost training speed.",
        },
        "ru": {
            "label": "Ускоритель",
            "info": "Подход, используемый для ускорения обучения.",
        },
        "zh": {"label": "加速方式", "info": "使用的加速方法。"},
        "ko": {
            "label": "부스터",
            "info": "훈련 속도를 향상시키기 위해 사용된 접근 방식.",
        },
        "ja": {
            "label": "ブースター",
            "info": "トレーニング速度を向上させるためのアプローチ。",
        },
    },
    "training_stage": {
        "en": {
            "label": "Stage",
            "info": "The stage to perform in training.",
        },
        "ru": {
            "label": "Этап",
            "info": "Этап выполнения обучения.",
        },
        "zh": {
            "label": "训练阶段",
            "info": "目前采用的训练方式。",
        },
        "ko": {
            "label": "학습 단계",
            "info": "수행할 학습 방법.",
        },
        "ja": {
            "label": "ステージ",
            "info": "トレーニングで実行するステージ。",
        },
    },
    "dataset_dir": {
        "en": {
            "label": "Data dir",
            "info": "Path to the data directory.",
        },
        "ru": {
            "label": "Директория данных",
            "info": "Путь к директории данных.",
        },
        "zh": {
            "label": "数据路径",
            "info": "数据文件夹的路径。",
        },
        "ko": {
            "label": "데이터 디렉토리",
            "info": "데이터 디렉토리의 경로.",
        },
        "ja": {
            "label": "データディレクトリ",
            "info": "データディレクトリへのパス。",
        },
    },
    "dataset": {
        "en": {
            "label": "Dataset",
        },
        "ru": {
            "label": "Набор данных",
        },
        "zh": {
            "label": "数据集",
        },
        "ko": {
            "label": "데이터셋",
        },
        "ja": {
            "label": "データセット",
        },
    },
    "data_preview_btn": {
        "en": {
            "value": "Preview dataset",
        },
        "ru": {
            "value": "Просмотреть набор данных",
        },
        "zh": {
            "value": "预览数据集",
        },
        "ko": {
            "value": "데이터셋 미리보기",
        },
        "ja": {
            "value": "データセットをプレビュー",
        },
    },
    "preview_count": {
        "en": {
            "label": "Count",
        },
        "ru": {
            "label": "Количество",
        },
        "zh": {
            "label": "数量",
        },
        "ko": {
            "label": "개수",
        },
        "ja": {
            "label": "カウント",
        },
    },
    "page_index": {
        "en": {
            "label": "Page",
        },
        "ru": {
            "label": "Страница",
        },
        "zh": {
            "label": "页数",
        },
        "ko": {
            "label": "페이지",
        },
        "ja": {
            "label": "ページ",
        },
    },
    "prev_btn": {
        "en": {
            "value": "Prev",
        },
        "ru": {
            "value": "Предыдущая",
        },
        "zh": {
            "value": "上一页",
        },
        "ko": {
            "value": "이전",
        },
        "ja": {
            "value": "前へ",
        },
    },
    "next_btn": {
        "en": {
            "value": "Next",
        },
        "ru": {
            "value": "Следующая",
        },
        "zh": {
            "value": "下一页",
        },
        "ko": {
            "value": "다음",
        },
        "ja": {
            "value": "次へ",
        },
    },
    "close_btn": {
        "en": {
            "value": "Close",
        },
        "ru": {
            "value": "Закрыть",
        },
        "zh": {
            "value": "关闭",
        },
        "ko": {
            "value": "닫기",
        },
        "ja": {
            "value": "閉じる",
        },
    },
    "preview_samples": {
        "en": {
            "label": "Samples",
        },
        "ru": {
            "label": "Примеры",
        },
        "zh": {
            "label": "样例",
        },
        "ko": {
            "label": "샘플",
        },
        "ja": {
            "label": "サンプル",
        },
    },
    "learning_rate": {
        "en": {
            "label": "Learning rate",
            "info": "Initial learning rate for AdamW.",
        },
        "ru": {
            "label": "Скорость обучения",
            "info": "Начальная скорость обучения для AdamW.",
        },
        "zh": {
            "label": "学习率",
            "info": "AdamW 优化器的初始学习率。",
        },
        "ko": {
            "label": "학습률",
            "info": "AdamW의 초기 학습률.",
        },
        "ja": {
            "label": "学習率",
            "info": "AdamW の初期学習率。",
        },
    },
    "num_train_epochs": {
        "en": {
            "label": "Epochs",
            "info": "Total number of training epochs to perform.",
        },
        "ru": {
            "label": "Эпохи",
            "info": "Общее количество эпох обучения.",
        },
        "zh": {
            "label": "训练轮数",
            "info": "需要执行的训练总轮数。",
        },
        "ko": {
            "label": "에포크",
            "info": "수행할 총 학습 에포크 수.",
        },
        "ja": {
            "label": "エポック数",
            "info": "実行するトレーニングの総エポック数。",
        },
    },
    "max_grad_norm": {
        "en": {
            "label": "Maximum gradient norm",
            "info": "Norm for gradient clipping.",
        },
        "ru": {
            "label": "Максимальная норма градиента",
            "info": "Норма для обрезки градиента.",
        },
        "zh": {
            "label": "最大梯度范数",
            "info": "用于梯度裁剪的范数。",
        },
        "ko": {
            "label": "최대 그레디언트 노름(norm)",
            "info": "그레디언트 클리핑을 위한 노름(norm).",
        },
        "ja": {
            "label": "最大勾配ノルム",
            "info": "勾配クリッピングのためのノルム。",
        },
    },
    "max_samples": {
        "en": {
            "label": "Max samples",
            "info": "Maximum samples per dataset.",
        },
        "ru": {
            "label": "Максимальное количество образцов",
            "info": "Максимальное количество образцов на набор данных.",
        },
        "zh": {
            "label": "最大样本数",
            "info": "每个数据集的最大样本数。",
        },
        "ko": {
            "label": "최대 샘플 수",
            "info": "데이터셋 당 최대 샘플 수.",
        },
        "ja": {
            "label": "最大サンプル数",
            "info": "データセットごとの最大サンプル数。",
        },
    },
    "compute_type": {
        "en": {
            "label": "Compute type",
            "info": "Whether to use mixed precision training.",
        },
        "ru": {
            "label": "Тип вычислений",
            "info": "Использовать ли обучение смешанной точности.",
        },
        "zh": {
            "label": "计算类型",
            "info": "是否使用混合精度训练。",
        },
        "ko": {
            "label": "연산 유형",
            "info": "혼합 정밀도 훈련을 사용할지 여부.",
        },
        "ja": {
            "label": "計算タイプ",
            "info": "混合精度トレーニングを使用するかどうか。",
        },
    },
    "compute_pscp": {
        "en": {
            "label": "Compute PSCP",
            "info": "Whether or not to compute the PSPC metric at evaluation.",
        },
        "ru": {
            "label": "Compute PSCP",
            "info": "Whether or not to compute the PSPC metric at evaluation.",
        },
        "zh": {
            "label": "Compute PSCP",
            "info": "Whether or not to compute the PSPC metric at evaluation.",
        },
        "ko": {
            "label": "Compute PSCP",
            "info": "Whether or not to compute the PSPC metric at evaluation.",
        },
        "ja": {
            "label": "Compute PSCP",
            "info": "Whether or not to compute the PSPC metric at evaluation.",
        },
    },
    "pscp_tab": {
        "en": {
            "label": "PSCP Settings",
        },
        "ru": {
            "label": "PSCP Settings",
        },
        "zh": {
            "label": "PSCP Settings",
        },
        "ko": {
            "label": "PSCP Settings",
        },
        "ja": {
            "label": "PSCP Settings",
        },
    },
    "pscp_memory": {
        "en": {
            "label": "PSCP Memory (GB)",
            "info": "Memory used during training.",
        },
        "ru": {
            "label": "PSCP Memory (GB)",
            "info": "Memory used during training.",
        },
        "zh": {
            "label": "PSCP Memory (GB)",
            "info": "Memory used during training.",
        },
        "ko": {
            "label": "PSCP Memory (GB)",
            "info": "Memory used during training.",
        },
        "ja": {
            "label": "PSCP Memory (GB)",
            "info": "Memory used during training.",
        },
    },
    "pscp_cp": {
        "en": {
            "label": "PSCP Cp",
            "info": "Cp constant for trainable parameters PSCP cost computation.",
        },
        "ru": {
            "label": "PSCP Cp",
            "info": "Cp constant for trainable parameters PSCP cost computation.",
        },
        "zh": {
            "label": "PSCP Cp",
            "info": "Cp constant for trainable parameters PSCP cost computation.",
        },
        "ko": {
            "label": "PSCP Cp",
            "info": "Cp constant for trainable parameters PSCP cost computation.",
        },
        "ja": {
            "label": "PSCP Cp",
            "info": "Cp constant for trainable parameters PSCP cost computation.",
        },
    },
    "pscp_cf": {
        "en": {
            "label": "PSCP Cf",
            "info": "Cf constant for inference time PSCP cost computation.",
        },
        "ru": {
            "label": "PSCP Cf",
            "info": "Cf constant for inference time PSCP cost computation.",
        },
        "zh": {
            "label": "PSCP Cf",
            "info": "Cf constant for inference time PSCP cost computation.",
        },
        "ko": {
            "label": "PSCP Cf",
            "info": "Cf constant for inference time PSCP cost computation.",
        },
        "ja": {
            "label": "PSCP Cf",
            "info": "Cf constant for inference time PSCP cost computation.",
        },
    },
    "pscp_cm": {
        "en": {
            "label": "PSCP Cm",
            "info": "Cm constant for memory factor in PSCP cost computation.",
        },
        "ru": {
            "label": "PSCP Cm",
            "info": "Cm constant for memory factor in PSCP cost computation.",
        },
        "zh": {
            "label": "PSCP Cm",
            "info": "Cm constant for memory factor in PSCP cost computation.",
        },
        "ko": {
            "label": "PSCP Cm",
            "info": "Cm constant for memory factor in PSCP cost computation.",
        },
        "ja": {
            "label": "PSCP Cm",
            "info": "Cm constant for memory factor in PSCP cost computation.",
        },
    },
    "pscp_bp": {
        "en": {
            "label": "PSCP Bp",
            "info": "Beta importance of paramters factor.",
        },
        "ru": {
            "label": "PSCP Bp",
            "info": "Beta importance of paramters factor.",
        },
        "zh": {
            "label": "PSCP Bp",
            "info": "Beta importance of paramters factor.",
        },
        "ko": {
            "label": "PSCP Bp",
            "info": "Beta importance of paramters factor.",
        },
        "ja": {
            "label": "PSCP Bp",
            "info": "Beta importance of paramters factor.",
        },
    },
    "pscp_bf": {
        "en": {
            "label": "PSCP Bf",
            "info": "Beta importance of inference time factor.",
        },
        "ru": {
            "label": "PSCP Bf",
            "info": "Beta importance of inference time factor.",
        },
        "zh": {
            "label": "PSCP Bf",
            "info": "Beta importance of inference time factor.",
        },
        "ko": {
            "label": "PSCP Bf",
            "info": "Beta importance of inference time factor.",
        },
        "ja": {
            "label": "PSCP Bf",
            "info": "Beta importance of inference time factor.",
        },
    },
    "pscp_bm": {
        "en": {
            "label": "PSCP Bm",
            "info": "Beta importance of memory factor.",
        },
        "ru": {
            "label": "PSCP Bm",
            "info": "Beta importance of memory factor.",
        },
        "zh": {
            "label": "PSCP Bm",
            "info": "Beta importance of memory factor.",
        },
        "ko": {
            "label": "PSCP Bm",
            "info": "Beta importance of memory factor.",
        },
        "ja": {
            "label": "PSCP Bm",
            "info": "Beta importance of memory factor.",
        },
    },
    "compute_classification_metrics": {
        "en": {
            "label": "Compute Classification Metrics",
            "info": "Whether or not to compute classification metrics at evaluation.",
        },
        "ru": {
            "label": "Compute Classification Metrics",
            "info": "Whether or not to compute classification metrics at evaluation.",
        },
        "zh": {
            "label": "Compute Classification Metrics",
            "info": "Whether or not to compute classification metrics at evaluation.",
        },
        "ko": {
            "label": "Compute Classification Metrics",
            "info": "Whether or not to compute classification metrics at evaluation.",
        },
        "ja": {
            "label": "Compute Classification Metrics",
            "info": "Whether or not to compute classification metrics at evaluation.",
        },
    },
    "cutoff_len": {
        "en": {
            "label": "Cutoff length",
            "info": "Max tokens in input sequence.",
        },
        "ru": {
            "label": "Длина обрезки",
            "info": "Максимальное количество токенов во входной последовательности.",
        },
        "zh": {
            "label": "截断长度",
            "info": "输入序列分词后的最大长度。",
        },
        "ko": {
            "label": "컷오프 길이",
            "info": "입력 시퀀스의 최대 토큰 수.",
        },
        "ja": {
            "label": "カットオフ長",
            "info": "入力シーケンスの最大トークン数。",
        },
    },
    "batch_size": {
        "en": {
            "label": "Batch size",
            "info": "Number of samples processed on each GPU.",
        },
        "ru": {
            "label": "Размер пакета",
            "info": "Количество образцов для обработки на каждом GPU.",
        },
        "zh": {
            "label": "批处理大小",
            "info": "每个 GPU 处理的样本数量。",
        },
        "ko": {
            "label": "배치 크기",
            "info": "각 GPU에서 처리되는 샘플 수.",
        },
        "ja": {
            "label": "バッチサイズ",
            "info": "各 GPU で処理されるサンプル数。",
        },
    },
    "gradient_accumulation_steps": {
        "en": {
            "label": "Gradient accumulation",
            "info": "Number of steps for gradient accumulation.",
        },
        "ru": {
            "label": "Накопление градиента",
            "info": "Количество шагов накопления градиента.",
        },
        "zh": {
            "label": "梯度累积",
            "info": "梯度累积的步数。",
        },
        "ko": {
            "label": "그레디언트 누적",
            "info": "그레디언트 누적 단계 수.",
        },
        "ja": {
            "label": "勾配累積",
            "info": "勾配累積のステップ数。",
        },
    },
    "val_size": {
        "en": {
            "label": "Val size",
            "info": "Percentage of validation set from the entire dataset.",
        },
        "ru": {
            "label": "Размер валидации",
            "info": "Пропорция данных в наборе для разработки.",
        },
        "zh": {
            "label": "验证集比例",
            "info": "验证集占全部样本的百分比。",
        },
        "ko": {
            "label": "검증 데이터셋 크기",
            "info": "개발 데이터셋에서 검증 데이터의 비율.",
        },
        "ja": {
            "label": "検証セットサイズ",
            "info": "データセット全体に対する検証セットの割合。",
        },
    },
    "lr_scheduler_type": {
        "en": {
            "label": "LR scheduler",
            "info": "Name of the learning rate scheduler.",
        },
        "ru": {
            "label": "Планировщик скорости обучения",
            "info": "Название планировщика скорости обучения.",
        },
        "zh": {
            "label": "学习率调节器",
            "info": "学习率调度器的名称。",
        },
        "ko": {
            "label": "LR 스케줄러",
            "info": "학습률 스케줄러의 이름.",
        },
        "ja": {
            "label": "学習率スケジューラ",
            "info": "学習率スケジューラの名前。",
        },
    },
    "extra_tab": {
        "en": {
            "label": "Extra configurations",
        },
        "ru": {
            "label": "Дополнительные конфигурации",
        },
        "zh": {
            "label": "其它参数设置",
        },
        "ko": {
            "label": "추가 구성(configuration)",
        },
        "ja": {
            "label": "追加設定",
        },
    },
    "logging_steps": {
        "en": {
            "label": "Logging steps",
            "info": "Number of steps between two logs.",
        },
        "ru": {
            "label": "Шаги логирования",
            "info": "Количество шагов между двумя записями в журнале.",
        },
        "zh": {
            "label": "日志间隔",
            "info": "每两次日志输出间的更新步数。",
        },
        "ko": {
            "label": "로깅 스텝",
            "info": "이전 로깅과 다음 로깅 간 스텝 수.",
        },
        "ja": {
            "label": "ロギングステップ",
            "info": "2 つのログ間のステップ数。",
        },
    },
    "save_steps": {
        "en": {
            "label": "Save steps",
            "info": "Number of steps between two checkpoints.",
        },
        "ru": {
            "label": "Шаги сохранения",
            "info": "Количество шагов между двумя контрольными точками.",
        },
        "zh": {
            "label": "保存间隔",
            "info": "每两次断点保存间的更新步数。",
        },
        "ko": {
            "label": "저장 스텝",
            "info": "이전 체크포인트와 다음 체크포인트 사이의 스텝 수.",
        },
        "ja": {
            "label": "保存ステップ",
            "info": "2 つのチェックポイント間のステップ数。",
        },
    },
    "warmup_steps": {
        "en": {
            "label": "Warmup steps",
            "info": "Number of steps used for warmup.",
        },
        "ru": {
            "label": "Шаги прогрева",
            "info": "Количество шагов, используемых для прогрева.",
        },
        "zh": {
            "label": "预热步数",
            "info": "学习率预热采用的步数。",
        },
        "ko": {
            "label": "Warmup 스텝",
            "info": "Warmup에 사용되는 스텝 수.",
        },
        "ja": {
            "label": "ウォームアップステップ",
            "info": "ウォームアップに使用されるステップ数。",
        },
    },
    "neftune_alpha": {
        "en": {
            "label": "NEFTune alpha",
            "info": "Magnitude of noise adding to embedding vectors.",
        },
        "ru": {
            "label": "NEFTune alpha",
            "info": "Величина шума, добавляемого к векторам вложений.",
        },
        "zh": {
            "label": "NEFTune 噪声参数",
            "info": "嵌入向量所添加的噪声大小。",
        },
        "ko": {
            "label": "NEFTune 알파",
            "info": "임베딩 벡터에 추가되는 노이즈의 크기.",
        },
        "ja": {
            "label": "NEFTune alpha",
            "info": "埋め込みベクトルに追加されるノイズの大きさ。",
        },
    },
    "extra_args": {
        "en": {
            "label": "Extra arguments",
            "info": "Extra arguments passed to the trainer in JSON format.",
        },
        "ru": {
            "label": "Дополнительные аргументы",
            "info": "Дополнительные аргументы, которые передаются тренеру в формате JSON.",
        },
        "zh": {
            "label": "额外参数",
            "info": "以 JSON 格式传递给训练器的额外参数。",
        },
        "ko": {
            "label": "추가 인수",
            "info": "JSON 형식으로 트레이너에게 전달할 추가 인수입니다.",
        },
        "ja": {
            "label": "追加引数",
            "info": "JSON 形式でトレーナーに渡される追加引数。",
        },
    },
    "packing": {
        "en": {
            "label": "Pack sequences",
            "info": "Pack sequences into samples of fixed length.",
        },
        "ru": {
            "label": "Упаковка последовательностей",
            "info": "Упаковка последовательностей в образцы фиксированной длины.",
        },
        "zh": {
            "label": "序列打包",
            "info": "将序列打包为等长样本。",
        },
        "ko": {
            "label": "시퀀스 패킹",
            "info": "고정된 길이의 샘플로 시퀀스를 패킹합니다.",
        },
        "ja": {
            "label": "シーケンスパッキング",
            "info": "シーケンスを固定長のサンプルにパッキングします。",
        },
    },
    "neat_packing": {
        "en": {
            "label": "Use neat packing",
            "info": "Avoid cross-attention between packed sequences.",
        },
        "ru": {
            "label": "Используйте аккуратную упаковку",
            "info": "избегайте перекрестного внимания между упакованными последовательностями.",
        },
        "zh": {
            "label": "使用无污染打包",
            "info": "避免打包后的序列产生交叉注意力。",
        },
        "ko": {
            "label": "니트 패킹 사용",
            "info": "패킹된 시퀀스 간의 크로스 어텐션을 피합니다.",
        },
        "ja": {
            "label": "無汚染パッキングを使用",
            "info": "パッキング後のシーケンス間のクロスアテンションを避けます。",
        },
    },
    "train_on_prompt": {
        "en": {
            "label": "Train on prompt",
            "info": "Disable the label mask on the prompt (only for SFT).",
        },
        "ru": {
            "label": "Тренировка на подсказке",
            "info": "Отключить маску меток на подсказке (только для SFT).",
        },
        "zh": {
            "label": "学习提示词",
            "info": "不在提示词的部分添加掩码（仅适用于 SFT）。",
        },
        "ko": {
            "label": "프롬프트도 학습",
            "info": "프롬프트에서 라벨 마스킹을 비활성화합니다 (SFT에만 해당).",
        },
        "ja": {
            "label": "プロンプトで学習",
            "info": "プロンプト部分にマスクを追加しない（SFT のみ）。",
        },
    },
    "mask_history": {
        "en": {
            "label": "Mask history",
            "info": "Train on the last turn only (only for SFT).",
        },
        "ru": {
            "label": "История масок",
            "info": "Тренироваться только на последнем шаге (только для SFT).",
        },
        "zh": {
            "label": "不学习历史对话",
            "info": "仅学习最后一轮对话（仅适用于 SFT）。",
        },
        "ko": {
            "label": "히스토리 마스킹",
            "info": "대화 데이터의 마지막 턴만 학습합니다 (SFT에만 해당).",
        },
        "ja": {
            "label": "履歴をマスク",
            "info": "最後のターンのみを学習する（SFT のみ）。",
        },
    },
    "resize_vocab": {
        "en": {
            "label": "Resize token embeddings",
            "info": "Resize the tokenizer vocab and the embedding layers.",
        },
        "ru": {
            "label": "Изменение размера токенных эмбеддингов",
            "info": "Изменить размер словаря токенизатора и слоев эмбеддинга.",
        },
        "zh": {
            "label": "更改词表大小",
            "info": "更改分词器词表和嵌入层的大小。",
        },
        "ko": {
            "label": "토큰 임베딩의 사이즈 조정",
            "info": "토크나이저 어휘와 임베딩 레이어의 크기를 조정합니다.",
        },
        "ja": {
            "label": "トークン埋め込みのサイズ変更",
            "info": "トークナイザーの語彙と埋め込み層のサイズを変更します。",
        },
    },
    "use_llama_pro": {
        "en": {
            "label": "Enable LLaMA Pro",
            "info": "Make the parameters in the expanded blocks trainable.",
        },
        "ru": {
            "label": "Включить LLaMA Pro",
            "info": "Сделать параметры в расширенных блоках обучаемыми.",
        },
        "zh": {
            "label": "使用 LLaMA Pro",
            "info": "仅训练块扩展后的参数。",
        },
        "ko": {
            "label": "LLaMA Pro 사용",
            "info": "확장된 블록의 매개변수를 학습 가능하게 만듭니다.",
        },
        "ja": {
            "label": "LLaMA Pro を有効化",
            "info": "拡張ブロックのパラメータのみをトレーニングします。",
        },
    },
    "enable_thinking": {
        "en": {
            "label": "Enable thinking",
            "info": "Whether or not to enable thinking mode for reasoning models.",
        },
        "ru": {
            "label": "Включить мысли",
            "info": "Включить режим мысли для моделей решающего характера.",
        },
        "zh": {
            "label": "启用思考模式",
            "info": "是否启用推理模型的思考模式。",
        },
        "ko": {
            "label": "생각 모드 활성화",
            "info": "추론 모델의 생각 모드를 활성화할지 여부.",
        },
        "ja": {
            "label": "思考モードを有効化",
            "info": "推論モデルの思考モードを有効にするかどうか。",
        },
    },
    "report_to": {
        "en": {
            "label": "Enable external logger",
            "info": "Use TensorBoard or wandb to log experiment.",
        },
        "ru": {
            "label": "Включить внешний регистратор",
            "info": "Использовать TensorBoard или wandb для ведения журнала экспериментов.",
        },
        "zh": {
            "label": "启用外部记录面板",
            "info": "使用 TensorBoard 或 wandb 记录实验。",
        },
        "ko": {
            "label": "외부 logger 활성화",
            "info": "TensorBoard 또는 wandb를 사용하여 실험을 기록합니다.",
        },
        "ja": {
            "label": "外部ロガーを有効化",
            "info": "TensorBoard または wandb を使用して実験を記録します。",
        },
    },
    "freeze_tab": {
        "en": {
            "label": "Freeze tuning configurations",
        },
        "ru": {
            "label": "конфигурации для настройки заморозки",
        },
        "zh": {
            "label": "部分参数微调设置",
        },
        "ko": {
            "label": "Freeze tuning 설정",
        },
        "ja": {
            "label": "フリーズチューニング設定",
        },
    },
    "freeze_trainable_layers": {
        "en": {
            "label": "Trainable layers",
            "info": "Number of the last(+)/first(-) hidden layers to be set as trainable.",
        },
        "ru": {
            "label": "Обучаемые слои",
            "info": "Количество последних (+)/первых (-) скрытых слоев, которые будут установлены как обучаемые.",
        },
        "zh": {
            "label": "可训练层数",
            "info": "最末尾（+）/最前端（-）可训练隐藏层的数量。",
        },
        "ko": {
            "label": "학습 가능한 레이어",
            "info": "학습 가능하게 설정할 마지막(+)/처음(-) 히든 레이어의 수.",
        },
        "ja": {
            "label": "学習可能なレイヤー",
            "info": "最後（+）/最初（-）の学習可能な隠れ層の数。",
        },
    },
    "freeze_trainable_modules": {
        "en": {
            "label": "Trainable modules",
            "info": "Name(s) of trainable modules. Use commas to separate multiple modules.",
        },
        "ru": {
            "label": "Обучаемые модули",
            "info": "Название обучаемых модулей. Используйте запятые для разделения нескольких модулей.",
        },
        "zh": {
            "label": "可训练模块",
            "info": "可训练模块的名称。使用英文逗号分隔多个名称。",
        },
        "ko": {
            "label": "학습 가능한 모듈",
            "info": "학습 가능한 모듈의 이름. 여러 모듈을 구분하려면 쉼표(,)를 사용하세요.",
        },
        "ja": {
            "label": "学習可能なモジュール",
            "info": "学習可能なモジュールの名前。複数のモジュールを区切るにはカンマを使用します。",
        },
    },
    "freeze_extra_modules": {
        "en": {
            "label": "Extra modules (optional)",
            "info": (
                "Name(s) of modules apart from hidden layers to be set as trainable. "
                "Use commas to separate multiple modules."
            ),
        },
        "ru": {
            "label": "Дополнительные модули (опционально)",
            "info": (
                "Имена модулей, кроме скрытых слоев, которые следует установить в качестве обучаемых. "
                "Используйте запятые для разделения нескольких модулей."
            ),
        },
        "zh": {
            "label": "额外模块（非必填）",
            "info": "除隐藏层以外的可训练模块名称。使用英文逗号分隔多个名称。",
        },
        "ko": {
            "label": "추가 모듈 (선택 사항)",
            "info": "히든 레이어 외에 학습 가능하게 설정할 모듈의 이름. 모듈 간에는 쉼표(,)로 구분하십시오.",
        },
        "ja": {
            "label": "追加モジュール（オプション）",
            "info": "隠れ層以外の学習可能なモジュールの名前。複数のモジュールを区切るにはカンマを使用します。",
        },
    },
    "lora_tab": {
        "en": {
            "label": "LoRA configurations",
        },
        "ru": {
            "label": "Конфигурации LoRA",
        },
        "zh": {
            "label": "LoRA 参数设置",
        },
        "ko": {
            "label": "LoRA 구성",
        },
        "ja": {
            "label": "LoRA 設定",
        },
    },
    "lora_rank": {
        "en": {
            "label": "LoRA rank",
            "info": "The rank of LoRA matrices.",
        },
        "ru": {
            "label": "Ранг матриц LoRA",
            "info": "Ранг матриц LoRA.",
        },
        "zh": {
            "label": "LoRA 秩",
            "info": "LoRA 矩阵的秩大小。",
        },
        "ko": {
            "label": "LoRA 랭크",
            "info": "LoRA 행렬의 랭크.",
        },
        "ja": {
            "label": "LoRA ランク",
            "info": "LoRA 行列のランク。",
        },
    },
    "lora_alpha": {
        "en": {
            "label": "LoRA alpha",
            "info": "Lora scaling coefficient.",
        },
        "ru": {
            "label": "LoRA alpha",
            "info": "Коэффициент масштабирования LoRA.",
        },
        "zh": {
            "label": "LoRA 缩放系数",
            "info": "LoRA 缩放系数大小。",
        },
        "ko": {
            "label": "LoRA 알파",
            "info": "LoRA 스케일링 계수.",
        },
        "ja": {
            "label": "LoRA alpha",
            "info": "LoRA スケーリング係数。",
        },
    },
    "lora_dropout": {
        "en": {
            "label": "LoRA dropout",
            "info": "Dropout ratio of LoRA weights.",
        },
        "ru": {
            "label": "Вероятность отсева LoRA",
            "info": "Вероятность отсева весов LoRA.",
        },
        "zh": {
            "label": "LoRA 随机丢弃",
            "info": "LoRA 权重随机丢弃的概率。",
        },
        "ko": {
            "label": "LoRA 드롭아웃",
            "info": "LoRA 가중치의 드롭아웃 비율.",
        },
        "ja": {
            "label": "LoRA ドロップアウト",
            "info": "LoRA 重みのドロップアウト確率。",
        },
    },
    "loraplus_lr_ratio": {
        "en": {
            "label": "LoRA+ LR ratio",
            "info": "The LR ratio of the B matrices in LoRA.",
        },
        "ru": {
            "label": "LoRA+ LR коэффициент",
            "info": "Коэффициент LR матриц B в LoRA.",
        },
        "zh": {
            "label": "LoRA+ 学习率比例",
            "info": "LoRA+ 中 B 矩阵的学习率倍数。",
        },
        "ko": {
            "label": "LoRA+ LR 비율",
            "info": "LoRA에서 B 행렬의 LR 비율.",
        },
        "ja": {
            "label": "LoRA+ LR 比率",
            "info": "LoRA+ の B 行列の学習率倍率。",
        },
    },
    "create_new_adapter": {
        "en": {
            "label": "Create new adapter",
            "info": "Create a new adapter with randomly initialized weight upon the existing one.",
        },
        "ru": {
            "label": "Создать новый адаптер",
            "info": "Создать новый адаптер с случайной инициализацией веса на основе существующего.",
        },
        "zh": {
            "label": "新建适配器",
            "info": "在现有的适配器上创建一个随机初始化后的新适配器。",
        },
        "ko": {
            "label": "새 어댑터 생성",
            "info": "기존 어댑터 위에 무작위로 초기화된 가중치를 가진 새 어댑터를 생성합니다.",
        },
        "ja": {
            "label": "新しいアダプターを作成",
            "info": "既存のアダプター上にランダムに初期化された新しいアダプターを作成します。",
        },
    },
    "use_rslora": {
        "en": {
            "label": "Use rslora",
            "info": "Use the rank stabilization scaling factor for LoRA layer.",
        },
        "ru": {
            "label": "Использовать rslora",
            "info": "Использовать коэффициент масштабирования стабилизации ранга для слоя LoRA.",
        },
        "zh": {
            "label": "使用 rslora",
            "info": "对 LoRA 层使用秩稳定缩放方法。",
        },
        "ko": {
            "label": "rslora 사용",
            "info": "LoRA 레이어에 랭크 안정화 스케일링 계수를 사용합니다.",
        },
        "ja": {
            "label": "rslora を使用",
            "info": "LoRA 層にランク安定化スケーリング方法を使用します。",
        },
    },
    "use_dora": {
        "en": {
            "label": "Use DoRA",
            "info": "Use weight-decomposed LoRA.",
        },
        "ru": {
            "label": "Используйте DoRA",
            "info": "Используйте LoRA с декомпозицией весов.",
        },
        "zh": {
            "label": "使用 DoRA",
            "info": "使用权重分解的 LoRA。",
        },
        "ko": {
            "label": "DoRA 사용",
            "info": "가중치-분해 LoRA를 사용합니다.",
        },
        "ja": {
            "label": "DoRA を使用",
            "info": "重み分解された LoRA を使用します。",
        },
    },
    "use_pissa": {
        "en": {
            "label": "Use PiSSA",
            "info": "Use PiSSA method.",
        },
        "ru": {
            "label": "используйте PiSSA",
            "info": "Используйте метод PiSSA.",
        },
        "zh": {
            "label": "使用 PiSSA",
            "info": "使用 PiSSA 方法。",
        },
        "ko": {
            "label": "PiSSA 사용",
            "info": "PiSSA 방법을 사용합니다.",
        },
        "ja": {
            "label": "PiSSA を使用",
            "info": "PiSSA メソッドを使用します。",
        },
    },
    "lora_target": {
        "en": {
            "label": "LoRA modules (optional)",
            "info": "Name(s) of modules to apply LoRA. Use commas to separate multiple modules.",
        },
        "ru": {
            "label": "Модули LoRA (опционально)",
            "info": "Имена модулей для применения LoRA. Используйте запятые для разделения нескольких модулей.",
        },
        "zh": {
            "label": "LoRA 作用模块（非必填）",
            "info": "应用 LoRA 的模块名称。使用英文逗号分隔多个名称。",
        },
        "ko": {
            "label": "LoRA 모듈 (선택 사항)",
            "info": "LoRA를 적용할 모듈의 이름. 모듈 간에는 쉼표(,)로 구분하십시오.",
        },
        "ja": {
            "label": "LoRA モジュール（オプション）",
            "info": "LoRA を適用するモジュールの名前。複数のモジュールを区切るにはカンマを使用します。",
        },
    },
    "additional_target": {
        "en": {
            "label": "Additional modules (optional)",
            "info": (
                "Name(s) of modules apart from LoRA layers to be set as trainable. "
                "Use commas to separate multiple modules."
            ),
        },
        "ru": {
            "label": "Дополнительные модули (опционально)",
            "info": (
                "Имена модулей, кроме слоев LoRA, которые следует установить в качестве обучаемых. "
                "Используйте запятые для разделения нескольких модулей."
            ),
        },
        "zh": {
            "label": "附加模块（非必填）",
            "info": "除 LoRA 层以外的可训练模块名称。使用英文逗号分隔多个名称。",
        },
        "ko": {
            "label": "추가 모듈 (선택 사항)",
            "info": "LoRA 레이어 외에 학습 가능하게 설정할 모듈의 이름. 모듈 간에는 쉼표(,)로 구분하십시오.",
        },
        "ja": {
            "label": "追加モジュール（オプション）",
            "info": "LoRA 層以外の学習可能なモジュールの名前。複数のモジュールを区切るにはカンマを使用します。",
        },
    },
    "peft_tab": {
        "en": {
            "label": "PEFT configurations",
        },
        "ru": {
            "label": "PEFT configurations",
        },
        "zh": {
            "label": "PEFT configurations",
        },
        "ko": {
            "label": "PEFT configurations",
        },
        "ja": {
            "label": "PEFT configurations",
        },
    },
    "task_type": {
        "en": {
            "label": "Task type",
            "info": "Set training task type for PEFT",
        },
        "ru": {
            "label": "Task type",
            "info": "Set training task type for PEFT",
        },
        "zh": {
            "label": "Task type",
            "info": "Set training task type for PEFT",
        },
        "ko": {
            "label": "Task type",
            "info": "Set training task type for PEFT",
        },
        "ja": {
            "label": "Task type",
            "info": "Set training task type for PEFT",
        },
    },
    "inference_mode": {
        "en": {
            "label": "Inference mode",
            "info": "Set PEFT adapter to inference mode",
        },
        "ru": {
            "label": "Inference mode",
            "info": "Set PEFT adapter to inference mode",
        },
        "zh": {
            "label": "Inference mode",
            "info": "Set PEFT adapter to inference mode",
        },
        "ko": {
            "label": "Inference mode",
            "info": "Set PEFT adapter to inference mode",
        },
        "ja": {
            "label": "Inference mode",
            "info": "Set PEFT adapter to inference mode",
        },
    },
    "rlhf_tab": {
        "en": {
            "label": "RLHF configurations",
        },
        "ru": {
            "label": "Конфигурации RLHF",
        },
        "zh": {
            "label": "RLHF 参数设置",
        },
        "ko": {
            "label": "RLHF 구성",
        },
        "ja": {
            "label": "RLHF 設定",
        },
    },
    "pref_beta": {
        "en": {
            "label": "Beta value",
            "info": "Value of the beta parameter in the loss.",
        },
        "ru": {
            "label": "Бета значение",
            "info": "Значение параметра бета в функции потерь.",
        },
        "zh": {
            "label": "Beta 参数",
            "info": "损失函数中 beta 超参数大小。",
        },
        "ko": {
            "label": "베타 값",
            "info": "손실 함수에서 베타 매개 변수의 값.",
        },
        "ja": {
            "label": "Beta 値",
            "info": "損失関数における beta ハイパーパラメータの値。",
        },
    },
    "pref_ftx": {
        "en": {
            "label": "Ftx gamma",
            "info": "The weight of SFT loss in the final loss.",
        },
        "ru": {
            "label": "Ftx гамма",
            "info": "Вес потери SFT в итоговой потере.",
        },
        "zh": {
            "label": "Ftx gamma",
            "info": "损失函数中 SFT 损失的权重大小。",
        },
        "ko": {
            "label": "Ftx 감마",
            "info": "최종 로스 함수에서 SFT 로스의 가중치.",
        },
        "ja": {
            "label": "Ftx gamma",
            "info": "損失関数における SFT 損失の重み。",
        },
    },
    "pref_loss": {
        "en": {
            "label": "Loss type",
            "info": "The type of the loss function.",
        },
        "ru": {
            "label": "Тип потерь",
            "info": "Тип функции потерь.",
        },
        "zh": {
            "label": "损失类型",
            "info": "损失函数的类型。",
        },
        "ko": {
            "label": "로스 유형",
            "info": "로스 함수의 유형.",
        },
        "ja": {
            "label": "損失タイプ",
            "info": "損失関数のタイプ。",
        },
    },
    "reward_model": {
        "en": {
            "label": "Reward model",
            "info": "Adapter of the reward model in PPO training.",
        },
        "ru": {
            "label": "Модель вознаграждения",
            "info": "Адаптер модели вознаграждения для обучения PPO.",
        },
        "zh": {
            "label": "奖励模型",
            "info": "PPO 训练中奖励模型的适配器路径。",
        },
        "ko": {
            "label": "리워드 모델",
            "info": "PPO 학습에서 사용할 리워드 모델의 어댑터.",
        },
        "ja": {
            "label": "報酬モデル",
            "info": "PPO トレーニングにおける報酬モデルのアダプター。",
        },
    },
    "ppo_score_norm": {
        "en": {
            "label": "Score norm",
            "info": "Normalizing scores in PPO training.",
        },
        "ru": {
            "label": "Норма оценок",
            "info": "Нормализация оценок в тренировке PPO.",
        },
        "zh": {
            "label": "归一化分数",
            "info": "PPO 训练中归一化奖励分数。",
        },
        "ko": {
            "label": "스코어 정규화",
            "info": "PPO 학습에서 스코어를 정규화합니다.",
        },
        "ja": {
            "label": "スコア正規化",
            "info": "PPO トレーニングにおける報酬スコアの正規化。",
        },
    },
    "ppo_whiten_rewards": {
        "en": {
            "label": "Whiten rewards",
            "info": "Whiten the rewards in PPO training.",
        },
        "ru": {
            "label": "Белые вознаграждения",
            "info": "Осветлите вознаграждения в обучении PPO.",
        },
        "zh": {
            "label": "白化奖励",
            "info": "PPO 训练中将奖励分数做白化处理。",
        },
        "ko": {
            "label": "보상 백화",
            "info": "PPO 훈련에서 보상을 백화(Whiten)합니다.",
        },
        "ja": {
            "label": "報酬のホワイトニング",
            "info": "PPO トレーニングにおいて報酬スコアをホワイトニング処理します。",
        },
    },
    "mm_tab": {
        "en": {
            "label": "Multimodal configurations",
        },
        "ru": {
            "label": "Конфигурации мультимедиа",
        },
        "zh": {
            "label": "多模态参数设置",
        },
        "ko": {
            "label": "멀티모달 구성",
        },
        "ja": {
            "label": "多モーダル設定",
        },
    },
    "freeze_vision_tower": {
        "en": {
            "label": "Freeze vision tower",
            "info": "Freeze the vision tower in the model.",
        },
        "ru": {
            "label": "Заморозить башню визиона",
            "info": "Заморозить башню визиона в модели.",
        },
        "zh": {
            "label": "冻结视觉编码器",
            "info": "冻结模型中的视觉编码器。",
        },
        "ko": {
            "label": "비전 타워 고정",
            "info": "모델의 비전 타워를 고정합니다.",
        },
        "ja": {
            "label": "ビジョンタワーの固定",
            "info": "モデルのビジョンタワーを固定します。",
        },
    },
    "freeze_multi_modal_projector": {
        "en": {
            "label": "Freeze multi-modal projector",
            "info": "Freeze the multi-modal projector in the model.",
        },
        "ru": {
            "label": "Заморозить мультимодальный проектор",
            "info": "Заморозить мультимодальный проектор в модели.",
        },
        "zh": {
            "label": "冻结多模态投影器",
            "info": "冻结模型中的多模态投影器。",
        },
        "ko": {
            "label": "멀티모달 프로젝터 고정",
            "info": "모델의 멀티모달 프로젝터를 고정합니다.",
        },
        "ja": {
            "label": "多モーダルプロジェクターの固定",
            "info": "モデルの多モーダルプロジェクターを固定します。",
        },
    },
    "freeze_language_model": {
        "en": {
            "label": "Freeze language model",
            "info": "Freeze the language model in the model.",
        },
        "ru": {
            "label": "Заморозить язык модели",
            "info": "Заморозить язык модели в модели.",
        },
        "zh": {
            "label": "冻结语言模型",
            "info": "冻结模型中的语言模型。",
        },
        "ko": {
            "label": "언어 모델 고정",
            "info": "모델의 언어 모델을 고정합니다.",
        },
        "ja": {
            "label": "言語モデルの固定",
            "info": "モデルの言語モデルを固定します。",
        },
    },
    "image_max_pixels": {
        "en": {
            "label": "Image max pixels",
            "info": "The maximum number of pixels of image inputs.",
        },
        "ru": {
            "label": "Максимальное количество пикселей изображения",
            "info": "Максимальное количество пикселей изображения.",
        },
        "zh": {
            "label": "图像最大像素",
            "info": "输入图像的最大像素数。",
        },
        "ko": {
            "label": "이미지 최대 픽셀",
            "info": "이미지 입력의 최대 픽셀 수입니다.",
        },
        "ja": {
            "label": "画像最大ピクセル",
            "info": "画像入力の最大ピクセル数です。",
        },
    },
    "image_min_pixels": {
        "en": {
            "label": "Image min pixels",
            "info": "The minimum number of pixels of image inputs.",
        },
        "ru": {
            "label": "Минимальное количество пикселей изображения",
            "info": "Минимальное количество пикселей изображения.",
        },
        "zh": {
            "label": "图像最小像素",
            "info": "输入图像的最小像素数。",
        },
        "ko": {
            "label": "이미지 최소 픽셀",
            "info": "이미지 입력의 최소 픽셀 수입니다.",
        },
        "ja": {
            "label": "画像最小ピクセル",
            "info": "画像入力の最小ピクセル数です。",
        },
    },
    "video_max_pixels": {
        "en": {
            "label": "Video max pixels",
            "info": "The maximum number of pixels of video inputs.",
        },
        "ru": {
            "label": "Максимальное количество пикселей видео",
            "info": "Максимальное количество пикселей видео.",
        },
        "zh": {
            "label": "视频最大像素",
            "info": "输入视频的最大像素数。",
        },
        "ko": {
            "label": "비디오 최대 픽셀",
            "info": "비디오 입력의 최대 픽셀 수입니다.",
        },
        "ja": {
            "label": "ビデオ最大ピクセル",
            "info": "ビデオ入力の最大ピクセル数です。",
        },
    },
    "video_min_pixels": {
        "en": {
            "label": "Video min pixels",
            "info": "The minimum number of pixels of video inputs.",
        },
        "ru": {
            "label": "Минимальное количество пикселей видео",
            "info": "Минимальное количество пикселей видео.",
        },
        "zh": {
            "label": "视频最小像素",
            "info": "输入视频的最小像素数。",
        },
        "ko": {
            "label": "비디오 최소 픽셀",
            "info": "비디오 입력의 최소 픽셀 수입니다.",
        },
        "ja": {
            "label": "ビデオ最小ピクセル",
            "info": "ビデオ入力の最小ピクセル数です。",
        },
    },
    "galore_tab": {
        "en": {
            "label": "GaLore configurations",
        },
        "ru": {
            "label": "Конфигурации GaLore",
        },
        "zh": {
            "label": "GaLore 参数设置",
        },
        "ko": {
            "label": "GaLore 구성",
        },
        "ja": {
            "label": "GaLore 設定",
        },
    },
    "use_galore": {
        "en": {
            "label": "Use GaLore",
            "info": "Use [GaLore](https://github.com/jiaweizzhao/GaLore) optimizer.",
        },
        "ru": {
            "label": "Использовать GaLore",
            "info": "Используйте оптимизатор [GaLore](https://github.com/jiaweizzhao/GaLore).",
        },
        "zh": {
            "label": "使用 GaLore",
            "info": "使用 [GaLore](https://github.com/jiaweizzhao/GaLore) 优化器。",
        },
        "ko": {
            "label": "GaLore 사용",
            "info": "[GaLore](https://github.com/jiaweizzhao/GaLore) 최적화를 사용하세요.",
        },
        "ja": {
            "label": "GaLore を使用",
            "info": "[GaLore](https://github.com/jiaweizzhao/GaLore) オプティマイザーを使用します。",
        },
    },
    "galore_rank": {
        "en": {
            "label": "GaLore rank",
            "info": "The rank of GaLore gradients.",
        },
        "ru": {
            "label": "Ранг GaLore",
            "info": "Ранг градиентов GaLore.",
        },
        "zh": {
            "label": "GaLore 秩",
            "info": "GaLore 梯度的秩大小。",
        },
        "ko": {
            "label": "GaLore 랭크",
            "info": "GaLore 그레디언트의 랭크.",
        },
        "ja": {
            "label": "GaLore ランク",
            "info": "GaLore 勾配のランク。",
        },
    },
    "galore_update_interval": {
        "en": {
            "label": "Update interval",
            "info": "Number of steps to update the GaLore projection.",
        },
        "ru": {
            "label": "Интервал обновления",
            "info": "Количество шагов для обновления проекции GaLore.",
        },
        "zh": {
            "label": "更新间隔",
            "info": "相邻两次投影更新的步数。",
        },
        "ko": {
            "label": "업데이트 간격",
            "info": "GaLore 프로젝션을 업데이트할 간격의 스텝 수.",
        },
        "ja": {
            "label": "更新間隔",
            "info": "隣接する 2 回の投影更新間のステップ数。",
        },
    },
    "galore_scale": {
        "en": {
            "label": "GaLore scale",
            "info": "GaLore scaling coefficient.",
        },
        "ru": {
            "label": "LoRA Alpha",
            "info": "Коэффициент масштабирования GaLore.",
        },
        "zh": {
            "label": "GaLore 缩放系数",
            "info": "GaLore 缩放系数大小。",
        },
        "ko": {
            "label": "GaLore 스케일",
            "info": "GaLore 스케일링 계수.",
        },
        "ja": {
            "label": "GaLore スケール",
            "info": "GaLore スケーリング係数。",
        },
    },
    "galore_target": {
        "en": {
            "label": "GaLore modules",
            "info": "Name(s) of modules to apply GaLore. Use commas to separate multiple modules.",
        },
        "ru": {
            "label": "Модули GaLore",
            "info": "Имена модулей для применения GaLore. Используйте запятые для разделения нескольких модулей.",
        },
        "zh": {
            "label": "GaLore 作用模块",
            "info": "应用 GaLore 的模块名称。使用英文逗号分隔多个名称。",
        },
        "ko": {
            "label": "GaLore 모듈",
            "info": "GaLore를 적용할 모듈의 이름. 모듈 간에는 쉼표(,)로 구분하십시오.",
        },
        "ja": {
            "label": "GaLore モジュール",
            "info": "GaLore を適用するモジュールの名前。複数のモジュールを区切るにはカンマを使用します。",
        },
    },
    "apollo_tab": {
        "en": {
            "label": "APOLLO configurations",
        },
        "ru": {
            "label": "Конфигурации APOLLO",
        },
        "zh": {
            "label": "APOLLO 参数设置",
        },
        "ko": {
            "label": "APOLLO 구성",
        },
        "ja": {
            "label": "APOLLO 設定",
        },
    },
    "use_apollo": {
        "en": {
            "label": "Use APOLLO",
            "info": "Use [APOLLO](https://github.com/zhuhanqing/APOLLO) optimizer.",
        },
        "ru": {
            "label": "Использовать APOLLO",
            "info": "Используйте оптимизатор [APOLLO](https://github.com/zhuhanqing/APOLLO).",
        },
        "zh": {
            "label": "使用 APOLLO",
            "info": "使用 [APOLLO](https://github.com/zhuhanqing/APOLLO) 优化器。",
        },
        "ko": {
            "label": "APOLLO 사용",
            "info": "[APOLLO](https://github.com/zhuhanqing/APOLLO) 최적화를 사용하세요.",
        },
        "ja": {
            "label": "APOLLO を使用",
            "info": "[APOLLO](https://github.com/zhuhanqing/APOLLO) オプティマイザーを使用します。",
        },
    },
    "apollo_rank": {
        "en": {
            "label": "APOLLO rank",
            "info": "The rank of APOLLO gradients.",
        },
        "ru": {
            "label": "Ранг APOLLO",
            "info": "Ранг градиентов APOLLO.",
        },
        "zh": {
            "label": "APOLLO 秩",
            "info": "APOLLO 梯度的秩大小。",
        },
        "ko": {
            "label": "APOLLO 랭크",
            "info": "APOLLO 그레디언트의 랭크.",
        },
        "ja": {
            "label": "APOLLO ランク",
            "info": "APOLLO 勾配のランク。",
        },
    },
    "apollo_update_interval": {
        "en": {
            "label": "Update interval",
            "info": "Number of steps to update the APOLLO projection.",
        },
        "ru": {
            "label": "Интервал обновления",
            "info": "Количество шагов для обновления проекции APOLLO.",
        },
        "zh": {
            "label": "更新间隔",
            "info": "相邻两次投影更新的步数。",
        },
        "ko": {
            "label": "업데이트 간격",
            "info": "APOLLO 프로젝션을 업데이트할 간격의 스텝 수.",
        },
        "ja": {
            "label": "更新間隔",
            "info": "隣接する 2 回の投影更新間のステップ数。",
        },
    },
    "apollo_scale": {
        "en": {
            "label": "APOLLO scale",
            "info": "APOLLO scaling coefficient.",
        },
        "ru": {
            "label": "LoRA Alpha",
            "info": "Коэффициент масштабирования APOLLO.",
        },
        "zh": {
            "label": "APOLLO 缩放系数",
            "info": "APOLLO 缩放系数大小。",
        },
        "ko": {
            "label": "APOLLO 스케일",
            "info": "APOLLO 스케일링 계수.",
        },
        "ja": {
            "label": "APOLLO スケール",
            "info": "APOLLO スケーリング係数。",
        },
    },
    "apollo_target": {
        "en": {
            "label": "APOLLO modules",
            "info": "Name(s) of modules to apply APOLLO. Use commas to separate multiple modules.",
        },
        "ru": {
            "label": "Модули APOLLO",
            "info": "Имена модулей для применения APOLLO. Используйте запятые для разделения нескольких модулей.",
        },
        "zh": {
            "label": "APOLLO 作用模块",
            "info": "应用 APOLLO 的模块名称。使用英文逗号分隔多个名称。",
        },
        "ko": {
            "label": "APOLLO 모듈",
            "info": "APOLLO를 적용할 모듈의 이름. 모듈 간에는 쉼표(,)로 구분하십시오.",
        },
        "ja": {
            "label": "APOLLO モジュール",
            "info": "APOLLO を適用するモジュールの名前。複数のモジュールを区切るにはカンマを使用します。",
        },
    },
    "badam_tab": {
        "en": {
            "label": "BAdam configurations",
        },
        "ru": {
            "label": "Конфигурации BAdam",
        },
        "zh": {
            "label": "BAdam 参数设置",
        },
        "ko": {
            "label": "BAdam 설정",
        },
        "ja": {
            "label": "BAdam 設定",
        },
    },
    "use_badam": {
        "en": {
            "label": "Use BAdam",
            "info": "Enable the [BAdam](https://github.com/Ledzy/BAdam) optimizer.",
        },
        "ru": {
            "label": "Использовать BAdam",
            "info": "Включите оптимизатор [BAdam](https://github.com/Ledzy/BAdam).",
        },
        "zh": {
            "label": "使用 BAdam",
            "info": "使用 [BAdam](https://github.com/Ledzy/BAdam) 优化器。",
        },
        "ko": {
            "label": "BAdam 사용",
            "info": "[BAdam](https://github.com/Ledzy/BAdam) 옵티마이저를 사용합니다.",
        },
        "ja": {
            "label": "BAdam を使用",
            "info": "[BAdam](https://github.com/Ledzy/BAdam) オプティマイザーを使用します。",
        },
    },
    "badam_mode": {
        "en": {
            "label": "BAdam mode",
            "info": "Whether to use layer-wise or ratio-wise BAdam optimizer.",
        },
        "ru": {
            "label": "Режим BAdam",
            "info": "Использовать ли оптимизатор BAdam с послоевой или пропорциональной настройкой.",
        },
        "zh": {
            "label": "BAdam 模式",
            "info": "使用 layer-wise 或 ratio-wise BAdam 优化器。",
        },
        "ko": {
            "label": "BAdam 모드",
            "info": "레이어-BAdam 옵티마이저인지 비율-BAdam 옵티마이저인지.",
        },
        "ja": {
            "label": "BAdam モード",
            "info": "layer-wise または ratio-wise BAdam オプティマイザーを使用します。",
        },
    },
    "badam_switch_mode": {
        "en": {
            "label": "Switch mode",
            "info": "The strategy of picking block to update for layer-wise BAdam.",
        },
        "ru": {
            "label": "Режим переключения",
            "info": "Стратегия выбора блока для обновления для послойного BAdam.",
        },
        "zh": {
            "label": "切换策略",
            "info": "Layer-wise BAdam 优化器的块切换策略。",
        },
        "ko": {
            "label": "스위치 모드",
            "info": "레이어-BAdam을 위한 블록 선택 전략.",
        },
        "ja": {
            "label": "切り替え戦略",
            "info": "Layer-wise BAdam オプティマイザーのブロック切り替え戦略。",
        },
    },
    "badam_switch_interval": {
        "en": {
            "label": "Switch interval",
            "info": "Number of steps to update the block for layer-wise BAdam.",
        },
        "ru": {
            "label": "Интервал переключения",
            "info": "количество шагов для обновления блока для пошагового BAdam.",
        },
        "zh": {
            "label": "切换频率",
            "info": "Layer-wise BAdam 优化器的块切换频率。",
        },
        "ko": {
            "label": "전환 간격",
            "info": "레이어-BAdam을 위한 블록 업데이트 간 스텝 수.",
        },
        "ja": {
            "label": "切り替え頻度",
            "info": "Layer-wise BAdam オプティマイザーのブロック切り替え頻度。",
        },
    },
    "badam_update_ratio": {
        "en": {
            "label": "Update ratio",
            "info": "The ratio of the update for ratio-wise BAdam.",
        },
        "ru": {
            "label": "Коэффициент обновления",
            "info": "Коэффициент обновления для BAdam с учётом соотношений.",
        },
        "zh": {
            "label": "Block 更新比例",
            "info": "Ratio-wise BAdam 优化器的更新比例。",
        },
        "ko": {
            "label": "업데이트 비율",
            "info": "비율-BAdam의 업데이트 비율.",
        },
        "ja": {
            "label": "ブロック更新比率",
            "info": "Ratio-wise BAdam オプティマイザーの更新比率。",
        },
    },
    "swanlab_tab": {
        "en": {
            "label": "SwanLab configurations",
        },
        "ru": {
            "label": "Конфигурации SwanLab",
        },
        "zh": {
            "label": "SwanLab 参数设置",
        },
        "ko": {
            "label": "SwanLab 설정",
        },
        "ja": {
            "label": "SwanLab 設定",
        },
    },
    "use_swanlab": {
        "en": {
            "label": "Use SwanLab",
            "info": "Enable [SwanLab](https://swanlab.cn/) for experiment tracking and visualization.",
        },
        "ru": {
            "label": "Использовать SwanLab",
            "info": "Включить [SwanLab](https://swanlab.cn/) для отслеживания и визуализации экспериментов.",
        },
        "zh": {
            "label": "使用 SwanLab",
            "info": "启用 [SwanLab](https://swanlab.cn/) 进行实验跟踪和可视化。",
        },
        "ko": {
            "label": "SwanLab 사용",
            "info": "[SwanLab](https://swanlab.cn/) 를 사용하여 실험을 추적하고 시각화합니다.",
        },
        "ja": {
            "label": "SwanLab を使用",
            "info": "[SwanLab](https://swanlab.cn/) を有効にして実験の追跡と可視化を行います。",
        },
    },
    "swanlab_project": {
        "en": {
            "label": "SwanLab project",
        },
        "ru": {
            "label": "SwanLab Проект",
        },
        "zh": {
            "label": "SwanLab 项目名",
        },
        "ko": {
            "label": "SwanLab 프로젝트",
        },
        "ja": {
            "label": "SwanLab プロジェクト",
        },
    },
    "swanlab_run_name": {
        "en": {
            "label": "SwanLab experiment name (optional)",
        },
        "ru": {
            "label": "SwanLab Имя эксперимента (опционально)",
        },
        "zh": {
            "label": "SwanLab 实验名（非必填）",
        },
        "ko": {
            "label": "SwanLab 실험 이름 (선택 사항)",
        },
        "ja": {
            "label": "SwanLab 実験名（オプション）",
        },
    },
    "swanlab_workspace": {
        "en": {
            "label": "SwanLab workspace (optional)",
            "info": "Workspace for SwanLab. Defaults to the personal workspace.",
        },
        "ru": {
            "label": "SwanLab Рабочая область (опционально)",
            "info": "Рабочая область SwanLab, если не заполнено, то по умолчанию в личной рабочей области.",
        },
        "zh": {
            "label": "SwanLab 工作区（非必填）",
            "info": "SwanLab 的工作区，默认在个人工作区下。",
        },
        "ko": {
            "label": "SwanLab 작업 영역 (선택 사항)",
            "info": "SwanLab 조직의 작업 영역, 비어 있으면 기본적으로 개인 작업 영역에 있습니다.",
        },
        "ja": {
            "label": "SwanLab ワークスペース（オプション）",
            "info": "SwanLab のワークスペース。デフォルトでは個人ワークスペースです。",
        },
    },
    "swanlab_api_key": {
        "en": {
            "label": "SwanLab API key (optional)",
            "info": "API key for SwanLab.",
        },
        "ru": {
            "label": "SwanLab API ключ (опционально)",
            "info": "API ключ для SwanLab.",
        },
        "zh": {
            "label": "SwanLab API 密钥（非必填）",
            "info": "用于在编程环境登录 SwanLab，已登录则无需填写。",
        },
        "ko": {
            "label": "SwanLab API 키 (선택 사항)",
            "info": "SwanLab의 API 키.",
        },
        "ja": {
            "label": "SwanLab API キー（オプション）",
            "info": "SwanLab の API キー。",
        },
    },
    "swanlab_mode": {
        "en": {
            "label": "SwanLab mode",
            "info": "Cloud or offline version.",
        },
        "ru": {
            "label": "SwanLab Режим",
            "info": "Версия в облаке или локальная версия.",
        },
        "zh": {
            "label": "SwanLab 模式",
            "info": "使用云端版或离线版 SwanLab。",
        },
        "ko": {
            "label": "SwanLab 모드",
            "info": "클라우드 버전 또는 오프라인 버전.",
        },
        "ja": {
            "label": "SwanLab モード",
            "info": "クラウド版またはオフライン版 SwanLab を使用します。",
        },
    },
    "swanlab_logdir": {
        "en": {
            "label": "SwanLab log directory",
            "info": "The log directory for SwanLab.",
        },
        "ru": {
            "label": "SwanLab 로그 디렉토리",
            "info": "SwanLab의 로그 디렉토리.",
        },
        "zh": {
            "label": "SwanLab 日志目录",
            "info": "SwanLab 的日志目录。",
        },
        "ko": {
            "label": "SwanLab 로그 디렉토리",
            "info": "SwanLab의 로그 디렉토리.",
        },
        "ja": {
            "label": "SwanLab ログ ディレクトリ",
            "info": "SwanLab のログ ディレクトリ。",
        },
    },
    "cmd_preview_btn": {
        "en": {
            "value": "Preview command",
        },
        "ru": {
            "value": "Просмотр команды",
        },
        "zh": {
            "value": "预览命令",
        },
        "ko": {
            "value": "명령어 미리보기",
        },
        "ja": {
            "value": "コマンドをプレビュー",
        },
    },
    "arg_save_btn": {
        "en": {
            "value": "Save arguments",
        },
        "ru": {
            "value": "Сохранить аргументы",
        },
        "zh": {
            "value": "保存训练参数",
        },
        "ko": {
            "value": "Argument 저장",
        },
        "ja": {
            "value": "引数を保存",
        },
    },
    "arg_load_btn": {
        "en": {
            "value": "Load arguments",
        },
        "ru": {
            "value": "Загрузить аргументы",
        },
        "zh": {
            "value": "载入训练参数",
        },
        "ko": {
            "value": "Argument 불러오기",
        },
        "ja": {
            "value": "引数を読み込む",
        },
    },
    "start_btn": {
        "en": {
            "value": "Start",
        },
        "ru": {
            "value": "Начать",
        },
        "zh": {
            "value": "开始",
        },
        "ko": {
            "value": "시작",
        },
        "ja": {
            "value": "開始",
        },
    },
    "stop_btn": {
        "en": {
            "value": "Abort",
        },
        "ru": {
            "value": "Прервать",
        },
        "zh": {
            "value": "中断",
        },
        "ko": {
            "value": "중단",
        },
        "ja": {
            "value": "中断",
        },
    },
    "output_dir": {
        "en": {
            "label": "Output dir",
            "info": "Directory for saving results.",
        },
        "ru": {
            "label": "Выходной каталог",
            "info": "Каталог для сохранения результатов.",
        },
        "zh": {
            "label": "输出目录",
            "info": "保存结果的路径。",
        },
        "ko": {
            "label": "출력 디렉토리",
            "info": "결과를 저장할 디렉토리.",
        },
        "ja": {
            "label": "出力ディレクトリ",
            "info": "結果を保存するパス。",
        },
    },
    "config_path": {
        "en": {
            "label": "Config path",
            "info": "Path to config saving arguments.",
        },
        "ru": {
            "label": "Путь к конфигурации",
            "info": "Путь для сохранения аргументов конфигурации.",
        },
        "zh": {
            "label": "配置路径",
            "info": "保存训练参数的配置文件路径。",
        },
        "ko": {
            "label": "설정 경로",
            "info": "Arguments 저장 파일 경로.",
        },
        "ja": {
            "label": "設定パス",
            "info": "トレーニングパラメータを保存する設定ファイルのパス。",
        },
    },
    "device_count": {
        "en": {
            "label": "Device count",
            "info": "Number of devices available.",
        },
        "ru": {
            "label": "Количество устройств",
            "info": "Количество доступных устройств.",
        },
        "zh": {
            "label": "设备数量",
            "info": "当前可用的运算设备数。",
        },
        "ko": {
            "label": "디바이스 수",
            "info": "사용 가능한 디바이스 수.",
        },
        "ja": {
            "label": "デバイス数",
            "info": "現在利用可能な演算デバイス数。",
        },
    },
    "ds_stage": {
        "en": {
            "label": "DeepSpeed stage",
            "info": "DeepSpeed stage for distributed training.",
        },
        "ru": {
            "label": "Этап DeepSpeed",
            "info": "Этап DeepSpeed для распределенного обучения.",
        },
        "zh": {
            "label": "DeepSpeed stage",
            "info": "多卡训练的 DeepSpeed stage。",
        },
        "ko": {
            "label": "DeepSpeed 단계",
            "info": "분산 학습을 위한 DeepSpeed 단계.",
        },
        "ja": {
            "label": "DeepSpeed stage",
            "info": "マルチ GPU トレーニングの DeepSpeed stage。",
        },
    },
    "ds_offload": {
        "en": {
            "label": "Enable offload",
            "info": "Enable DeepSpeed offload (slow down training).",
        },
        "ru": {
            "label": "Включить выгрузку",
            "info": "включить выгрузку DeepSpeed (замедлит обучение).",
        },
        "zh": {
            "label": "使用 offload",
            "info": "使用 DeepSpeed offload（会减慢速度）。",
        },
        "ko": {
            "label": "오프로딩 활성화",
            "info": "DeepSpeed 오프로딩 활성화 (훈련 속도 느려짐).",
        },
        "ja": {
            "label": "オフロードを使用",
            "info": "DeepSpeed オフロードを使用します（速度が遅くなります）。",
        },
    },
    "output_box": {
        "en": {
            "value": "Ready.",
        },
        "ru": {
            "value": "Готово.",
        },
        "zh": {
            "value": "准备就绪。",
        },
        "ko": {
            "value": "준비 완료.",
        },
        "ja": {
            "value": "準備完了。",
        },
    },
    "loss_viewer": {
        "en": {
            "label": "Loss",
        },
        "ru": {
            "label": "Потери",
        },
        "zh": {
            "label": "损失",
        },
        "ko": {
            "label": "손실",
        },
        "ja": {
            "label": "損失",
        },
    },
    "predict": {
        "en": {
            "label": "Save predictions",
        },
        "ru": {
            "label": "Сохранить предсказания",
        },
        "zh": {
            "label": "保存预测结果",
        },
        "ko": {
            "label": "예측 결과 저장",
        },
        "ja": {
            "label": "予測結果を保存",
        },
    },
    "infer_backend": {
        "en": {
            "label": "Inference engine",
        },
        "ru": {
            "label": "Инференс движок",
        },
        "zh": {
            "label": "推理引擎",
        },
        "ko": {
            "label": "추론 엔진",
        },
        "ja": {
            "label": "推論エンジン",
        },
    },
    "infer_dtype": {
        "en": {
            "label": "Inference data type",
        },
        "ru": {
            "label": "Тип данных для вывода",
        },
        "zh": {
            "label": "推理数据类型",
        },
        "ko": {
            "label": "추론 데이터 유형",
        },
        "ja": {
            "label": "推論データタイプ",
        },
    },
    "load_btn": {
        "en": {
            "value": "Load model",
        },
        "ru": {
            "value": "Загрузить модель",
        },
        "zh": {
            "value": "加载模型",
        },
        "ko": {
            "value": "모델 불러오기",
        },
        "ja": {
            "value": "モデルを読み込む",
        },
    },
    "unload_btn": {
        "en": {
            "value": "Unload model",
        },
        "ru": {
            "value": "Выгрузить модель",
        },
        "zh": {
            "value": "卸载模型",
        },
        "ko": {
            "value": "모델 언로드",
        },
        "ja": {
            "value": "モデルをアンロード",
        },
    },
    "info_box": {
        "en": {
            "value": "Model unloaded, please load a model first.",
        },
        "ru": {
            "value": "Модель не загружена, загрузите модель сначала.",
        },
        "zh": {
            "value": "模型未加载，请先加载模型。",
        },
        "ko": {
            "value": "모델이 언로드되었습니다. 모델을 먼저 불러오십시오.",
        },
        "ja": {
            "value": "モデルがロードされていません。最初にモデルをロードしてください。",
        },
    },
    "role": {
        "en": {
            "label": "Role",
        },
        "ru": {
            "label": "Роль",
        },
        "zh": {
            "label": "角色",
        },
        "ko": {
            "label": "역할",
        },
        "ja": {
            "label": "役割",
        },
    },
    "system": {
        "en": {
            "placeholder": "System prompt (optional)",
        },
        "ru": {
            "placeholder": "Системный запрос (по желанию)",
        },
        "zh": {
            "placeholder": "系统提示词（非必填）",
        },
        "ko": {
            "placeholder": "시스템 프롬프트 (선택 사항)",
        },
        "ja": {
            "placeholder": "システムプロンプト（オプション）",
        },
    },
    "tools": {
        "en": {
            "placeholder": "Tools (optional)",
        },
        "ru": {
            "placeholder": "Инструменты (по желанию)",
        },
        "zh": {
            "placeholder": "工具列表（非必填）",
        },
        "ko": {
            "placeholder": "툴 (선택 사항)",
        },
        "ja": {
            "placeholder": "ツールリスト（オプション）",
        },
    },
    "image": {
        "en": {
            "label": "Image (optional)",
        },
        "ru": {
            "label": "Изображение (по желанию)",
        },
        "zh": {
            "label": "图像（非必填）",
        },
        "ko": {
            "label": "이미지 (선택 사항)",
        },
        "ja": {
            "label": "画像（オプション）",
        },
    },
    "video": {
        "en": {
            "label": "Video (optional)",
        },
        "ru": {
            "label": "Видео (по желанию)",
        },
        "zh": {
            "label": "视频（非必填）",
        },
        "ko": {
            "label": "비디오 (선택 사항)",
        },
        "ja": {
            "label": "動画（オプション）",
        },
    },
    "query": {
        "en": {
            "placeholder": "Input...",
        },
        "ru": {
            "placeholder": "Ввод...",
        },
        "zh": {
            "placeholder": "输入...",
        },
        "ko": {
            "placeholder": "입력...",
        },
        "ja": {
            "placeholder": "入力...",
        },
    },
    "submit_btn": {
        "en": {
            "value": "Submit",
        },
        "ru": {
            "value": "Отправить",
        },
        "zh": {
            "value": "提交",
        },
        "ko": {
            "value": "제출",
        },
        "ja": {
            "value": "送信",
        },
    },
    "max_length": {
        "en": {
            "label": "Maximum length",
        },
        "ru": {
            "label": "Максимальная длина",
        },
        "zh": {
            "label": "最大长度",
        },
        "ko": {
            "label": "최대 길이",
        },
        "ja": {
            "label": "最大長",
        },
    },
    "max_new_tokens": {
        "en": {
            "label": "Maximum new tokens",
        },
        "ru": {
            "label": "Максимальное количество новых токенов",
        },
        "zh": {
            "label": "最大生成长度",
        },
        "ko": {
            "label": "응답의 최대 길이",
        },
        "ja": {
            "label": "最大生成長",
        },
    },
    "top_p": {
        "en": {
            "label": "Top-p",
        },
        "ru": {
            "label": "Лучшие-p",
        },
        "zh": {
            "label": "Top-p 采样值",
        },
        "ko": {
            "label": "Top-p",
        },
        "ja": {
            "label": "Top-p",
        },
    },
    "temperature": {
        "en": {
            "label": "Temperature",
        },
        "ru": {
            "label": "Температура",
        },
        "zh": {
            "label": "温度系数",
        },
        "ko": {
            "label": "온도",
        },
        "ja": {
            "label": "温度",
        },
    },
    "skip_special_tokens": {
        "en": {
            "label": "Skip special tokens",
        },
        "ru": {
            "label": "Пропустить специальные токены",
        },
        "zh": {
            "label": "跳过特殊 token",
        },
        "ko": {
            "label": "스페셜 토큰을 건너뛰기",
        },
        "ja": {
            "label": "スペシャルトークンをスキップ",
        },
    },
    "escape_html": {
        "en": {
            "label": "Escape HTML tags",
        },
        "ru": {
            "label": "Исключить HTML теги",
        },
        "zh": {
            "label": "转义 HTML 标签",
        },
        "ko": {
            "label": "HTML 태그 이스케이프",
        },
        "ja": {
            "label": "HTML タグをエスケープ",
        },
    },
    "clear_btn": {
        "en": {
            "value": "Clear history",
        },
        "ru": {
            "value": "Очистить историю",
        },
        "zh": {
            "value": "清空历史",
        },
        "ko": {
            "value": "기록 지우기",
        },
        "ja": {
            "value": "履歴をクリア",
        },
    },
    "export_size": {
        "en": {
            "label": "Max shard size (GB)",
            "info": "The maximum size for a model file.",
        },
        "ru": {
            "label": "Максимальный размер фрагмента (ГБ)",
            "info": "Максимальный размер файла модели.",
        },
        "zh": {
            "label": "最大分块大小（GB）",
            "info": "单个模型文件的最大大小。",
        },
        "ko": {
            "label": "최대 샤드 크기 (GB)",
            "info": "모델 파일의 최대 크기.",
        },
        "ja": {
            "label": "最大シャードサイズ（GB）",
            "info": "単一のモデルファイルの最大サイズ。",
        },
    },
    "export_quantization_bit": {
        "en": {
            "label": "Export quantization bit.",
            "info": "Quantizing the exported model.",
        },
        "ru": {
            "label": "Экспорт бита квантования",
            "info": "Квантование экспортируемой модели.",
        },
        "zh": {
            "label": "导出量化等级",
            "info": "量化导出模型。",
        },
        "ko": {
            "label": "양자화 비트 내보내기",
            "info": "내보낸 모델의 양자화.",
        },
        "ja": {
            "label": "量子化ビットをエクスポート",
            "info": "エクスポートするモデルを量子化します。",
        },
    },
    "export_quantization_dataset": {
        "en": {
            "label": "Export quantization dataset",
            "info": "The calibration dataset used for quantization.",
        },
        "ru": {
            "label": "Экспорт набора данных для квантования",
            "info": "Набор данных калибровки, используемый для квантования.",
        },
        "zh": {
            "label": "导出量化数据集",
            "info": "量化过程中使用的校准数据集。",
        },
        "ko": {
            "label": "양자화 데이터셋 내보내기",
            "info": "양자화에 사용되는 교정 데이터셋.",
        },
        "ja": {
            "label": "量子化データセットをエクスポート",
            "info": "量子化プロセスで使用されるキャリブレーションデータセット。",
        },
    },
    "export_device": {
        "en": {
            "label": "Export device",
            "info": "Which device should be used to export model.",
        },
        "ru": {
            "label": "Экспорт устройство",
            "info": "Какое устройство следует использовать для экспорта модели.",
        },
        "zh": {
            "label": "导出设备",
            "info": "导出模型使用的设备类型。",
        },
        "ko": {
            "label": "내보낼 장치",
            "info": "모델을 내보내는 데 사용할 장치.",
        },
        "ja": {
            "label": "エクスポートデバイス",
            "info": "モデルをエクスポートするために使用するデバイスタイプ。",
        },
    },
    "export_legacy_format": {
        "en": {
            "label": "Export legacy format",
            "info": "Do not use safetensors to save the model.",
        },
        "ru": {
            "label": "Экспорт в устаревший формат",
            "info": "Не использовать safetensors для сохранения модели.",
        },
        "zh": {
            "label": "导出旧格式",
            "info": "不使用 safetensors 格式保存模型。",
        },
        "ko": {
            "label": "레거시 형식 내보내기",
            "info": "모델을 저장하는 데 safetensors를 사용하지 않습니다.",
        },
        "ja": {
            "label": "レガシーフォーマットをエクスポート",
            "info": "safetensors フォーマットを使用せずにモデルを保存します。",
        },
    },
    "export_dir": {
        "en": {
            "label": "Export dir",
            "info": "Directory to save exported model.",
        },
        "ru": {
            "label": "Каталог экспорта",
            "info": "Каталог для сохранения экспортированной модели.",
        },
        "zh": {
            "label": "导出目录",
            "info": "保存导出模型的文件夹路径。",
        },
        "ko": {
            "label": "내보내기 디렉토리",
            "info": "내보낸 모델을 저장할 디렉토리.",
        },
        "ja": {
            "label": "エクスポートディレクトリ",
            "info": "エクスポートしたモデルを保存するフォルダのパス。",
        },
    },
    "export_hub_model_id": {
        "en": {
            "label": "HF Hub ID (optional)",
            "info": "Repo ID for uploading model to Hugging Face hub.",
        },
        "ru": {
            "label": "HF Hub ID (опционально)",
            "info": "Идентификатор репозитория для загрузки модели на Hugging Face hub.",
        },
        "zh": {
            "label": "HF Hub ID（非必填）",
            "info": "用于将模型上传至 Hugging Face Hub 的仓库 ID。",
        },
        "ko": {
            "label": "HF 허브 ID (선택 사항)",
            "info": "모델을 Hugging Face 허브에 업로드하기 위한 레포 ID.",
        },
        "ja": {
            "label": "HF Hub ID（オプション）",
            "info": "Hugging Face Hub にモデルをアップロードするためのリポジトリ ID。",
        },
    },
    "export_btn": {
        "en": {
            "value": "Export",
        },
        "ru": {
            "value": "Экспорт",
        },
        "zh": {
            "value": "开始导出",
        },
        "ko": {
            "value": "내보내기",
        },
        "ja": {
            "value": "エクスポート",
        },
    },
    "device_memory": {
        "en": {
            "label": "Device memory",
            "info": "Current memory usage of the device (GB).",
        },
        "ru": {
            "label": "Память устройства",
            "info": "Текущая память на устройстве (GB).",
        },
        "zh": {
            "label": "设备显存",
            "info": "当前设备的显存（GB）。",
        },
        "ko": {
            "label": "디바이스 메모리",
            "info": "지금 사용 중인 기기 메모리 (GB).",
        },
        "ja": {
            "label": "デバイスメモリ",
            "info": "現在のデバイスのメモリ（GB）。",
        },
    },
}


ALERTS = {
    "err_conflict": {
        "en": "A process is in running, please abort it first.",
        "ru": "Процесс уже запущен, пожалуйста, сначала прервите его.",
        "zh": "任务已存在，请先中断训练。",
        "ko": "프로세스가 실행 중입니다. 먼저 중단하십시오.",
        "ja": "プロセスが実行中です。最初に中断してください。",
    },
    "err_exists": {
        "en": "You have loaded a model, please unload it first.",
        "ru": "Вы загрузили модель, сначала разгрузите ее.",
        "zh": "模型已存在，请先卸载模型。",
        "ko": "모델이 로드되었습니다. 먼저 언로드하십시오.",
        "ja": "モデルがロードされています。最初にアンロードしてください。",
    },
    "err_no_model": {
        "en": "Please select a model.",
        "ru": "Пожалуйста, выберите модель.",
        "zh": "请选择模型。",
        "ko": "모델을 선택하십시오.",
        "ja": "モデルを選択してください。",
    },
    "err_no_path": {
        "en": "Model not found.",
        "ru": "Модель не найдена.",
        "zh": "模型未找到。",
        "ko": "모델을 찾을 수 없습니다.",
        "ja": "モデルが見つかりません。",
    },
    "err_no_dataset": {
        "en": "Please choose a dataset.",
        "ru": "Пожалуйста, выберите набор данных.",
        "zh": "请选择数据集。",
        "ko": "데이터 세트를 선택하십시오.",
        "ja": "データセットを選択してください。",
    },
    "err_no_adapter": {
        "en": "Please select an adapter.",
        "ru": "Пожалуйста, выберите адаптер.",
        "zh": "请选择适配器。",
        "ko": "어댑터를 선택하십시오.",
        "ja": "アダプターを選択してください。",
    },
    "err_no_output_dir": {
        "en": "Please provide output dir.",
        "ru": "Пожалуйста, укажите выходную директорию.",
        "zh": "请填写输出目录。",
        "ko": "출력 디렉토리를 제공하십시오.",
        "ja": "出力ディレクトリを入力してください。",
    },
    "err_no_reward_model": {
        "en": "Please select a reward model.",
        "ru": "Пожалуйста, выберите модель вознаграждения.",
        "zh": "请选择奖励模型。",
        "ko": "리워드 모델을 선택하십시오.",
        "ja": "報酬モデルを選択してください。",
    },
    "err_no_export_dir": {
        "en": "Please provide export dir.",
        "ru": "Пожалуйста, укажите каталог для экспорта.",
        "zh": "请填写导出目录。",
        "ko": "Export 디렉토리를 제공하십시오.",
        "ja": "エクスポートディレクトリを入力してください。",
    },
    "err_gptq_lora": {
        "en": "Please merge adapters before quantizing the model.",
        "ru": "Пожалуйста, объедините адаптеры перед квантованием модели.",
        "zh": "量化模型前请先合并适配器。",
        "ko": "모델을 양자화하기 전에 어댑터를 병합하십시오.",
        "ja": "モデルを量子化する前にアダプターをマージしてください。",
    },
    "err_failed": {
        "en": "Failed.",
        "ru": "Ошибка.",
        "zh": "训练出错。",
        "ko": "실패했습니다.",
        "ja": "失敗しました。",
    },
    "err_demo": {
        "en": "Training is unavailable in demo mode, duplicate the space to a private one first.",
        "ru": "Обучение недоступно в демонстрационном режиме, сначала скопируйте пространство в частное.",
        "zh": "展示模式不支持训练，请先复制到私人空间。",
        "ko": "데모 모드에서는 훈련을 사용할 수 없습니다. 먼저 프라이빗 레포지토리로 작업 공간을 복제하십시오.",
        "ja": "デモモードではトレーニングは利用できません。最初にプライベートスペースに複製してください。",
    },
    "err_tool_name": {
        "en": "Tool name not found.",
        "ru": "Имя инструмента не найдено.",
        "zh": "工具名称未找到。",
        "ko": "툴 이름을 찾을 수 없습니다.",
        "ja": "ツール名が見つかりません。",
    },
    "err_json_schema": {
        "en": "Invalid JSON schema.",
        "ru": "Неверная схема JSON.",
        "zh": "Json 格式错误。",
        "ko": "잘못된 JSON 스키마입니다.",
        "ja": "JSON スキーマが無効です。",
    },
    "err_config_not_found": {
        "en": "Config file is not found.",
        "ru": "Файл конфигурации не найден.",
        "zh": "未找到配置文件。",
        "ko": "Config 파일을 찾을 수 없습니다.",
        "ja": "設定ファイルが見つかりません。",
    },
    "warn_no_cuda": {
        "en": "CUDA environment was not detected.",
        "ru": "Среда CUDA не обнаружена.",
        "zh": "未检测到 CUDA 环境。",
        "ko": "CUDA 환경이 감지되지 않았습니다.",
        "ja": "CUDA 環境が検出されませんでした。",
    },
    "warn_output_dir_exists": {
        "en": "Output dir already exists, will resume training from here.",
        "ru": "Выходной каталог уже существует, обучение будет продолжено отсюда.",
        "zh": "输出目录已存在，将从该断点恢复训练。",
        "ko": "출력 디렉토리가 이미 존재합니다. 위 출력 디렉토리에 저장된 학습을 재개합니다.",
        "ja": "出力ディレクトリが既に存在します。このチェックポイントからトレーニングを再開します。",
    },
    "warn_no_instruct": {
        "en": "You are using a non-instruct model, please fine-tune it first.",
        "ru": "Вы используете модель без инструкции, пожалуйста, primeros выполните донастройку этой модели.",
        "zh": "您正在使用非指令模型，请先对其进行微调。",
        "ko": "당신은 지시하지 않은 모델을 사용하고 있습니다. 먼저 이를 미세 조정해 주세요.",
        "ja": "インストラクションモデルを使用していません。まずモデルをアダプターに適合させてください。",
    },
    "info_aborting": {
        "en": "Aborted, wait for terminating...",
        "ru": "Прервано, ожидание завершения...",
        "zh": "训练中断，正在等待进程结束……",
        "ko": "중단되었습니다. 종료를 기다리십시오...",
        "ja": "トレーニングが中断されました。プロセスの終了を待っています...",
    },
    "info_aborted": {
        "en": "Ready.",
        "ru": "Готово.",
        "zh": "准备就绪。",
        "ko": "준비되었습니다.",
        "ja": "準備完了。",
    },
    "info_finished": {
        "en": "Finished.",
        "ru": "Завершено.",
        "zh": "训练完毕。",
        "ko": "완료되었습니다.",
        "ja": "トレーニングが完了しました。",
    },
    "info_config_saved": {
        "en": "Arguments have been saved at: ",
        "ru": "Аргументы были сохранены по адресу: ",
        "zh": "训练参数已保存至：",
        "ko": "매개변수가 저장되었습니다: ",
        "ja": "トレーニングパラメータが保存されました: ",
    },
    "info_config_loaded": {
        "en": "Arguments have been restored.",
        "ru": "Аргументы были восстановлены.",
        "zh": "训练参数已载入。",
        "ko": "매개변수가 복원되었습니다.",
        "ja": "トレーニングパラメータが読み込まれました。",
    },
    "info_loading": {
        "en": "Loading model...",
        "ru": "Загрузка модели...",
        "zh": "加载中……",
        "ko": "모델 로딩 중...",
        "ja": "モデルをロード中...",
    },
    "info_unloading": {
        "en": "Unloading model...",
        "ru": "Выгрузка модели...",
        "zh": "卸载中……",
        "ko": "모델 언로딩 중...",
        "ja": "モデルをアンロード中...",
    },
    "info_loaded": {
        "en": "Model loaded, now you can chat with your model!",
        "ru": "Модель загружена, теперь вы можете общаться с вашей моделью!",
        "zh": "模型已加载，可以开始聊天了！",
        "ko": "모델이 로드되었습니다. 이제 모델과 채팅할 수 있습니다!",
        "ja": "モデルがロードされました。チャットを開始できます！",
    },
    "info_unloaded": {
        "en": "Model unloaded.",
        "ru": "Модель выгружена.",
        "zh": "模型已卸载。",
        "ko": "모델이 언로드되었습니다.",
        "ja": "モデルがアンロードされました。",
    },
    "info_thinking": {
        "en": "🌀 Thinking...",
        "ru": "🌀 Думаю...",
        "zh": "🌀 思考中...",
        "ko": "🌀 생각 중...",
        "ja": "🌀 考えています...",
    },
    "info_thought": {
        "en": "✅ Thought",
        "ru": "✅ Думать закончено",
        "zh": "✅ 思考完成",
        "ko": "✅ 생각이 완료되었습니다",
        "ja": "✅ 思考完了",
    },
    "info_exporting": {
        "en": "Exporting model...",
        "ru": "Экспорт модели...",
        "zh": "正在导出模型……",
        "ko": "모델 내보내기 중...",
        "ja": "モデルをエクスポート中...",
    },
    "info_exported": {
        "en": "Model exported.",
        "ru": "Модель экспортирована.",
        "zh": "模型导出完成。",
        "ko": "모델이 내보내졌습니다.",
        "ja": "モデルのエクスポートが完了しました。",
    },
    "info_swanlab_link": {
        "en": "### SwanLab Link\n",
        "ru": "### SwanLab ссылка\n",
        "zh": "### SwanLab 链接\n",
        "ko": "### SwanLab 링크\n",
        "ja": "### SwanLab リンク\n",
    },
}
