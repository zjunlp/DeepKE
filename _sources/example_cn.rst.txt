Example
=======

standard_ner
------------

进入 ``example/ner/standard`` 可使用全监督实体抽取模型。

在 ``data`` 文件夹下存放数据，训练等参数都在 ``conf`` 文件夹中，可修改使用。

运行 `run.py` 即可进行训练，运行 `predict.py` 即可进行预测。

.. code-block:: bash

    cd example/ner/standard

    python run.py
    
    python predict.py

few-shot_ner
------------

进入 ``example/ner/few-shot`` 可使用少样本实体抽取模型。

在 ``data`` 文件夹下存放数据，包含 `conll2003` ， `mit-movie` ， `mit-restaurant` 和 `atis` 等数据集。

训练等参数都在 ``conf`` 文件夹中，可修改使用。

运行 `python run.py` 即可对conll2003数据集进行训练，`python run.py +train=few_shot` 可进行少样本训练。

在 `config.yaml` 中加入 `- predict`  ， 再在 `predict.yaml` 中修改 `load_path` 为模型路径以及 `write_path` 为预测结果保存路径，再 `python predict.py` 可进行预测。


.. code-block:: bash

    cd example/ner/few-shot

    python run.py
    
    python predict.py

standard_re
-----------
进入 ``example/re/standard`` 可使用全监督关系抽取模型。

在 ``data/origin`` 文件夹下存放数据，训练等参数都在 ``conf`` 文件夹中，可修改使用。

运行 `python run.py` 即可进行训练，运行 `python redict.py` 即可进行预测。

.. code-block:: bash

    cd example/re/standard

    python run.py

    python predict.py

few-shot_re
-----------

进入 ``example/re/few-shot`` 可使用少样本关系抽取模型。

在 ``data`` 文件夹下存放数据，模型采用的数据集是SEMEVAL，SEMEVAL数据集来自于2010年的国际语义评测大会中 `Task 8："Multi-Way Classification of Semantic Relations Between Pairs of Nominals"` 。

训练等参数都在 ``conf`` 文件夹中，可修改使用。

运行 `python run.py` 即可进行训练,设置 ``conf`` 中 `train_from_saved_model` 为上次保存模型的路径即可从上次训练的模型开始训练

运行 `python predict.py` 即可进行预测。

.. code-block:: bash

    cd example/re/few-shot

    python run.py
    
    python predict.py

document_re
-----------

进入 ``example/re/document`` 可使用全监督关系抽取模型。

在 ``data`` 文件夹下存放数据，模型采用的数据集是 `DocRED` ，其中包含的 `train_distant.json` 由于文件太大，可自行从 https://drive.google.com/drive/folders/1c5-0YwnoJx8NS6CV2f-NoTHR__BdkNqw 上下载到data/目录下

训练等参数都在 ``conf`` 文件夹中，可修改使用。

运行 `python run.py` 即可进行训练，设置 ``conf`` 中 `train_from_saved_model` 为上次保存模型的路径即可从上次训练的模型开始训练。

运行 `python redict.py` 即可进行预测。最终生成的 `result.json` 文件保存在根目录。

.. code-block:: bash

    cd example/re/document

    python run.py

    python predict.py

standard_ae
-----------
进入 ``example/ae/standard`` 可使用全监督属性抽取模型。

在 ``data/origin`` 文件夹下存放数据，训练等参数都在 ``conf`` 文件夹中，可修改使用。

运行 `python run.py` 即可进行训练，运行 `python predict.py` 即可进行预测。

.. code-block:: bash

    cd example/ae/standard

    python run.py
    
    python predict.py


具体流程，请参考 https://www.bilibili.com/video/BV1n44y1x7iW?spm_id_from=333.999.0.0