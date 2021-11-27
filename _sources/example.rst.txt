Example
=======

Standard NER
------------
The standard module is implemented by the pretrained model BERT. 

**Step 1**

Enter  ``DeepKE/example/ner/standard`` .

**Step 2**

Get data: 

    `wget 120.27.214.45/Data/ner/standard/data.tar.gz`

    `tar -xzvf data.tar.gz`

The dataset and parameters can be customized in the ``data`` folder and ``conf`` folder respectively.

**Step 3**

Train:
    
     `python run.py`

**Step 4**

Predict:

     `python predict.py`

.. code-block:: bash

    cd example/ner/standard

    wget 120.27.214.45/Data/ner/standard/data.tar.gz

    tar -xzvf data.tar.gz

    python run.py
    
    python predict.py

Few-shot NER
------------
This module is in the low-resouce scenario.

**Step 1**

Enter  ``DeepKE/example/ner/few-shot`` . 

**Step 2**
Get data:

    `wget 120.27.214.45/Data/ner/few_shot/data.tar.gz`
    `tar -xzvf data.tar.gz`

The directory where the model is loaded and saved and the configuration parameters can be cusomized in the ``conf`` folder.The dataset can be customized in the ``data`` folder.

**Step 3**

Train with CoNLL-2003:

     `python run.py`

Train in the few-shot scenario: 
    
    `python run.py +train=few_shot`. Users can modify `load_path` in ``conf/train/few_shot.yaml`` with the use of existing loaded model.

**Step 4**

Predict: 
    
    add `- predict` to ``conf/config.yaml`` , modify `loda_path` as the model path and `write_path` as the path where the predicted results are saved in ``conf/predict.yaml`` , and then run `python predict.py`

.. code-block:: bash

    cd example/ner/few-shot

    wget 120.27.214.45/Data/ner/few_shot/data.tar.gz
    
    tar -xzvf data.tar.gz

    python run.py
    
    python predict.py

Standard RE
-----------
The standard module is implemented by common deep learning models, including CNN, RNN, Capsule, GCN, Transformer and the pretrained model.

**Step 1**

Enter the ``DeepKE/example/re/standard`` folder. 

**Step 2**

Get data:

    `wget 120.27.214.45/Data/re/standard/data.tar.gz`
    `tar -xzvf data.tar.gz`

The dataset and parameters can be customized in the ``data`` folder and ``conf`` folder respectively.

**Step 3**

Train:
    
     `python run.py`

**Step 4**

Predict:

     `python predict.py`

.. code-block:: bash

    cd example/re/standard

    wget 120.27.214.45/Data/re/standard/data.tar.gz

    tar -xzvf data.tar.gz

    python run.py

    python predict.py

Few-shot RE
-----------
This module is in the low-resouce scenario.

**Step 1**

Enter ``DeepKE/example/re/few-shot`` .

**Step 2**

Get data:

    `wget 120.27.214.45/Data/re/few_shot/data.tar.gz`
    `tar -xzvf data.tar.gz`

The dataset and parameters can be customized in the ``data`` folder and ``conf`` folder respectively.

**Step 3**

Train:
    
    `python run.py`

Start with the model trained last time: modify `train_from_saved_model` in ``conf/train.yaml`` as the path where the model trained last time was saved. And the path saving logs generated in training can be customized by ``log_dir``.

**Step 4**

Predict:

    `python predict.py`

.. code-block:: bash

    cd example/re/few-shot

    wget 120.27.214.45/Data/re/few_shot/data.tar.gz

    tar -xzvf data.tar.gz

    python run.py
    
    python predict.py

Document RE
-----------
This module is in the document scenario.

**Step 1**

Enter ``DeepKE/example/re/document`` .

**Step2**

Get data:

    `wget 120.27.214.45/Data/re/document/data.tar.gz`
    
    `tar -xzvf data.tar.gz`

The dataset and parameters can be customized in the ``data`` folder and ``conf`` folder respectively.

**Step 3**

Train:
    
    `python run.py`

Start with the model trained last time: modify `train_from_saved_model` in ``conf/train.yaml`` as the path where the model trained last time was saved. And the path saving logs generated in training can be customized by ``log_dir``.

**Step 4**

Predict:

    `python predict.py`

.. code-block:: bash

    cd example/re/document

    wget 120.27.214.45/Data/re/document/data.tar.gz

    tar -xzvf data.tar.gz

    python run.py

    python predict.py

Standard AE
-----------
The standard module is implemented by common deep learning models, including CNN, RNN, Capsule, GCN, Transformer and the pretrained model.

**Step 1**

Enter the ``DeepKE/example/ae/standard`` folder. 

**Step 2**

Get data:

    `wget 120.27.214.45/Data/ae/standard/data.tar.gz`

    `tar -xzvf data.tar.gz`

The dataset and parameters can be customized in the ``data`` folder and ``conf`` folder respectively.

**Step 3**

Train: 
    
    `python run.py`

**Step 4**

Predict:

    `python predict.py`

.. code-block:: bash

    cd example/ae/regular

    wget 120.27.214.45/Data/ae/standard/data.tar.gz

    tar -xzvf data.tar.gz

    python run.py

    python predict.py


More details , you can refer to https://www.bilibili.com/video/BV1n44y1x7iW?spm_id_from=333.999.0.0 .