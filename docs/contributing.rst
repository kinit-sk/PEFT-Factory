Contributing
============

In this section, you will find guidelines on how to contribute to the PEFT-Factory project, 
including adding new PEFT methods from supported providers, contributing datasets, and submitting pull requests.

.. _peft-providers-support:

PEFT Providers Support
---------------------

PEFT-Factory supports `HuggingFace PEFT <https://huggingface.co/docs/peft/index>`__ and `Adapters <https://adapterhub.ml/>`__ as PEFT providers. 
You can easily add new methods from these providers by extending the ``HF_PEFT_METHODS`` and ``ADAPTERS_METHODS``, 
as well as, their corresponding configuration mappings ``PEFT_CONFIG_MAPPING`` and ``ADAPTERS_CONFIG_MAPPING``.

Unlike the custom PEFT methods, PEFT provider methods require to be added by extending the constants in the codebase.
Therefore, new release is needed. Following instructions will guide you through the process of adding new PEFT methods from these providers.

Fork the repository
~~~~~~~~~~~~~~~~~~~

`Please fork and clone the repository on GitHub to make your changes. <https://github.com/kinit-sk/PEFT-Factory/fork>`__

Create new branch
~~~~~~~~~~~~~~~~~

First, create a new branch for your changes.

.. code:: bash

   git checkout -b add-new-peft-method


Update the constants file
~~~~~~~~~~~~~~~~~~~~~~~~~

The extension is done in ``src/llamafactory/extras/constants.py`` file.

For example, for adding new HuggingFace PEFT called ``FourierFT`` method the the before and after constants would look like


.. code:: python

    # Before adding the FourierFT PEFT method
    HF_PEFT_METHODS = ["prompt-tuning", "prefix-tuning", "p-tuning", "ia3", "lntuning", "mtp"]
    PEFT_CONFIG_MAPPING = {
        "prompt-tuning": PromptTuningConfig,
        "prefix-tuning": PrefixTuningConfig,
        "p-tuning": PromptEncoderConfig,
        "ia3": IA3Config,
        "lntuning": LNTuningConfig,
        "mtp": MultitaskPromptTuningConfig,
    }

.. code:: python

    # After adding the FourierFT PEFT method
    HF_PEFT_METHODS = ["prompt-tuning", "prefix-tuning", "p-tuning", "ia3", "lntuning", "mtp", "fourier-ft"]
    PEFT_CONFIG_MAPPING = {
        "prompt-tuning": PromptTuningConfig,
        "prefix-tuning": PrefixTuningConfig,
        "p-tuning": PromptEncoderConfig,
        "ia3": IA3Config,
        "lntuning": LNTuningConfig,
        "mtp": MultitaskPromptTuningConfig,
        "fourier-ft": FourierFTConfig,
    }


Commit nad push your changes.


Make a pull request
~~~~~~~~~~~~~~~~~~~~~~

After pushing your changes, go to your forked repository on GitHub and create a pull request to the main repository.
We will review your changes and merge them if everything is correct.


.. _adding-peft-methods:

Adding PEFT Methods
---------------------

To add new custom PEFT methods to PEFT-Factory, you can use our dynamic PEFT loading.
Basically, this loads the PEFT method config and model classes dynamically during runtinme from the PEFT directory.
The config and model classes are then added to the 

The PEFT directory can be set by environment variable ``PEFT_DIR`` with ``./peft`` directory as default.

To add a new PEFT method, create a new directory in the PEFT directory with the name of the PEFT method.

.. code:: bash

   mkdir -p peft/my_new_peft_method

For the dynamic loading to work, a ``model.py`` and ``config.py`` files are required in the new PEFT method directory.

In the model.py file, create a class that inherits from ``peft.tuners.tuners_utils.BaseTuner`` and implement the required methods.

In the config.py file, create a class that inherits from ``peft.PeftConfig`` and implement the required methods.

To get more information about the required methods and structure, you can refer to the existing PEFT methods in the PEFT directory or to the `HuggingFace PEFT repository <https://github.com/huggingface/peft>`__.

Templates
~~~~~~~~~

You can use following templates for creating the model and config classes.

config.py
^^^^^^^^

.. code:: python

    class CustomMethodConfig(PeftConfig):
        """Minimal configuration template for a custom PEFT method.

        Extend this class with any method-specific hyperparameters you need.
        """

        def __post_init__(self):
            super().__post_init__()
            if self.task_type is None:
                raise ValueError("task_type must be specified.")

model.py
^^^^^^^^

.. code:: python

    from typing import Optional

    import torch.nn as nn
    from peft import PeftConfig
    from peft.tuners.tuners_utils import BaseTuner


    class CustomMethodModel(BaseTuner):
        """Minimal model template for a custom PEFT tuner.

        Implement the adapter creation, forwarding logic, and optional merge/unmerge.
        """

        def __init__(self, base_model: nn.Module, config: PeftConfig):
            super().__init__(base_model=base_model, config=config)
            self.base_model = base_model
            self.config = config
            self.adapters = nn.ModuleDict()

        def _create_adapter(self, adapter_name: str) -> None:
            """Instantiate a new adapter and register it."""
            pass

        def set_adapter(self, adapter_name: str) -> None:
            """Set the active adapter for the model."""
            pass

        def forward(self, *args, **kwargs):
            """Define the forward pass with adapter logic (if required)."""
            pass

        def merge(self, adapter_names: Optional[list[str]] = None, safe_merge: bool = False) -> None:
            """Optional: Apply adapter weights into the base model."""
            return

        def unmerge(self) -> None:
            """Optional: Revert any merged adapters."""
            return
    
.. _adding-datasets:

Adding Datasets
--------------------

Adding new datasets for PEFT-Factory is similar to adding new datasets in LLaMA-Factory. 
On top of that, PEFT-Factory allows to add classification datasets with numerical classes and with a new field called instruction.

To add new dataset simply add a new item into the ``data/dataset_info.json`` file.

View following example:

.. code:: json

    "mrpc": {
        "hf_hub_url": "nyu-mll/glue",
        "subset": "mrpc",
        "split": "train",
        "instruction": "Classify the following sentence pair into not_equivalent and equivalent classes. Respond only with the corresponding class.",
        "columns": {
        "prompt": "sentence1",
        "query": "sentence2",
        "response": "label"
        }
    },
    
    