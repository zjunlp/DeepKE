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

Dataset needs to be input as ``TXT`` file

The `data's format` of file needs to comply with the following：

杭 B-LOC '\\n'
州 I-LOC '\\n'
真 O '\\n'
美 O '\\n'

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

Dataset needs to be input as ``TXT`` file

The `data's format` of file needs to comply with the following：

EU	B-ORG '\\n'
rejects	O '\\n'
German	B-MISC '\\n'
call	O '\\n'
to	O '\\n'
boycott	O '\\n'
British	B-MISC '\\n'
lamb	O '\\n'
.	O '\\n'

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

Multimodal NER
--------------
This module is in the multimodal scenario.

**Step 1**

Enter the ``DeepKE/example/ner/multimodal`` folder.

**Step 2**

Get data:

    `wget 120.27.214.45/Data/ner/multimodal/data.tar.gz`

    `tar -xzvf data.tar.gz`

The dataset and parameters can be customized in the ``data`` folder and ``conf`` folder respectively.

Textual dataset needs to be input as ``TXT`` file. Visual dataset is recommended to be input as ``JPG`` or ``PNG`` file.

The `data's format` of file needs to comply with the following：

IMGID:213820 '\\n'
RT	O '\\n'
@JonMatthewsDS	O '\\n'
:	O '\\n'
Driving	I-ORG '\\n'
test	O '\\n'
Leicester	B-ORG '\\n'
Jon	I-ORG '\\n'
Matthews	I-ORG '\\n'
Driving	I-ORG '\\n'
School	I-ORG '\\n'
:	O '\\n'
http://t.co/Zf0tOasjUE	O '\\n'
http://t.co/IXiIgDVmPu	O '\\n'

Instead of inputting the original images as visual datas directly, you can use a `Visual Grounding toolkit <https://github.com/zyang-ur/onestage_grounding>`_ to locate visual objects.

**Step 3**

Train:

    `python run.py`

Start with the model trained last time: modify `load_path` in ``conf/train.yaml`` as the path where the model trained last time was saved. And the path saving logs generated in training can be customized by ``log_dir``.

**Step 4**

Predict:

    `python predict.py`

.. code-block:: bash
    
    cd example/ner/multimodal

    wget 120.27.214.45/Data/ner/multimodal/data.tar.gz

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

Dataset needs to be input as ``CSV`` file.

The `data's format` of file needs to comply with the following：

+--------------------------+-----------+------------+-------------+------------+------------+
| Sentence                 | Relation  | Head       | Head_offset |  Tail      | Tail_offset|
+--------------------------+-----------+------------+-------------+------------+------------+

The relation's format of file needs to comply with the following：

+------------+-----------+------------------+-------------+
| Head_type  | Tail_type | relation         | Index       |
+------------+-----------+------------------+-------------+


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

Dataset needs to be input as ``TXT`` file and ``JSON`` file.

The `data's format` of file needs to comply with the following：

{"token": ["the", "most", "common", "audits", "were", "about", "waste", "and", "recycling", "."], "h": {"name": "audits", "pos": [3, 4]}, "t": {"name": "waste", "pos": [6, 7]}, "relation": "Message-Topic(e1,e2)"}

The relation's format of file needs to comply with the following：

{"Other": 0 , "Message-Topic(e1,e2)": 1 ... }

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


Dataset needs to be input as ``JSON`` file

The `data's format` of file needs to comply with the following：

[{"vertexSet": [[{"name": "Lark Force", "pos": [0, 2], "sent_id": 0, "type": "ORG"},...]], 

"labels": [{"r": "P607", "h": 1, "t": 3, "evidence": [0]}, ...], 

"title": "Lark Force",

"sents": [["Lark", "Force", "was", "an", "Australian", "Army", "formation", "established", "in", "March", "1941", "during", "World", "War", "II", "for", "service", "in", "New", "Britain", "and", "New", "Ireland", "."],...}]


The relation's format of file needs to comply with the following：

{"P1376": 79,"P607": 27,...}

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

Multimodal RE
-------------
This module is in the multimodal scenario.

**Step 1**

Enter ``DeepKE/example/re/multimodal`` .

**Step 2**

Get data:

    `wget 120.27.214.45/Data/re/multimodal/data.tar.gz`

    `tar -xzvf data.tar.gz`

The dataset and parameters can be customized in the ``data`` folder and ``conf`` folder respectively.

Textual dataset needs to be input as ``TXT`` file and ``JSON`` file. Visual dataset is recommended to be input as ``JPG`` or ``PNG`` file.

The `data's format` of file needs to comply with the following：

{'token': ['The', 'latest', 'Arkham', 'Horror', 'LCG', 'deluxe', 'expansion', 'the', 'Circle', 'Undone', 'has', 'been', 'released', ':'], 'h': {'name': 'Circle Undone', 'pos': [8, 10]}, 't': {'name': 'Arkham Horror LCG', 'pos': [2, 5]}, 'img_id': 'twitter_19_31_16_6.jpg', 'relation': '/misc/misc/part_of'}

The relation's format of file needs to comply with the following：

{"None":0,"/per/per/parent":1,"/per/per/siblings":2...}

Instead of inputting the original images as visual datas directly, you can use a `Visual Grounding toolkit <https://github.com/zyang-ur/onestage_grounding>`_ to locate visual objects based on entities and entity types.

**Step 3**

Train:

    `python run.py`

Start with the model trained last time: modify `load_path` in ``conf/train.yaml`` as the path where the model trained last time was saved. And the path saving logs generated in training can be customized by ``log_dir``.

**Step 4**

Predict:

    `python predict.py`

.. code-block:: bash

    cd example/re/multimodal

    wget 120.27.214.45/Data/re/multimodal/data.tar.gz

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

Dataset needs to be input as ``CSV`` file.

The `data's format` of file needs to comply with the following：

+--------------------------+------------+------------+---------------+-------------------+-----------------------+
| Sentence                 | Attribute  | Entity     | Entity_offset |  Attribute_value  | Attribute_value_offset|
+--------------------------+------------+------------+---------------+-------------------+-----------------------+

The attribute's format of file needs to comply with the following：

+-------------------+-------------+
| Attribute         | Index       |
+-------------------+-------------+

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