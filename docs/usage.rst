Usage
=====

.. _quickstart:

Quickstart
------------

For video example please visit the `PEFT-Factory Demonstration Video <https://www.youtube.com/watch?v=Q3kxvlyO-XY>`__.

.. code:: bash

   # install package
   pip install peftfactory

   # dowload repo that contains data, PEFT methods and examples
   git clone https://github.com/kinit-sk/PEFT-Factory.git && cd PEFT-Factory

   # start web UI
   pf webui


Alternatively, you can run training from command line:

.. code:: bash

   # install package
   pip install peftfactory

   # dowload repo that contains data, PEFT methods and examples
   git clone https://github.com/kinit-sk/PEFT-Factory.git && cd PEFT-Factory

Create some variables for envsubst
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # run training with config file
   TIMESTAMP=`date +%s`
   OUTPUT_DIR="saves/bitfit/llama-3.2-1b-instruct/train_wsc_${TIMESTAMP}"
   DATASET="wsc"
   SEED=123
   WANDB_PROJECT="peft-factory-train-bitfit"
   WANDB_NAME="bitfit_llama-3.2-1b-instruct_train_wsc"

   mkdir -p "${OUTPUT_DIR}"

   export OUTPUT_DIR DATASET SEED WANDB_PROJECT WANDB_NAME

Use the template
~~~~~~~~~~~~~~~~

Utility ``envsubst`` replaces the occurances of env variables with their values (see the template).

.. code:: bash

   envsubst < examples/peft/bitfit/llama-3.2-1b-instruct/train.yaml > ${OUTPUT_DIR}/train.yaml

Run the factory
~~~~~~~~~~~~~~~~
.. code:: bash

   peftfactory-cli train ${OUTPUT_DIR}/train.yaml

.. _installation:

Installation
------------

There are multiple ways to install PEFT-Factory. You can install develelopment version from source or install the latest release from PyPI.

Using pip
~~~~~~~~~~~~~
.. code:: bash

   pip install peftfactory

From Source
~~~~~~~~~~~~~

Clone the repository
^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   git clone git@github.com:kinit-sk/PEFT-Factory.git

Build the wheel package
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   make build

Install with pip
^^^^^^^^^^^^^^^^

.. code:: bash

   pip install dist/[name of the built package].whl

.. _get-data-and-methods:

Get data and methods
-------------------

To download data, methods and examples for training please download the repository from GitHub.

.. code:: bash

   git clone https://github.com/kinit-sk/PEFT-Factory.git && cd PEFT-Factory

.. _run-training:

Run training
------------

You can run training from command line or using web UI.

From Command Line
~~~~~~~~~~~~~~~~~~~~~~
To run training from command line use the following command:

.. code:: bash

   pf train [path to config file].yaml

Using web UI
~~~~~~~~~~~~~~~~

To run the web UI use the following command:
.. code:: bash

   pf webui

