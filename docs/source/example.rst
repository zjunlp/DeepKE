Example
=======

standard_ner
------------
- The standard module is implemented by the pretrained model BERT. 
- Enter  ``DeepKE/example/ner/standard`` .
- The dataset and parameters can be customized in the ``data`` folder and ``conf`` folder respectively.
- **Train**: `python run.py`
- **Predict**: `python predict.py`

.. code-block:: bash

    cd example/ner/standard

    python run.py
    
    python predict.py

few-shot_ner
------------
- This module is in the low-resouce scenario.
- Enter  ``DeepKE/example/ner/few-shot`` . 
- The directory where the model is loaded and saved and the configuration parameters can be cusomized in the ``conf`` folder.
- **Train with CoNLL-2003**: `python run.py`
- **Train in the few-shot scenario**: `python run.py +train=few_shot`. Users can modify `load_path` in ``conf/train/few_shot.yaml`` with the use of existing loaded model.
- **Predict**: add `- predict` to ``conf/config.yaml`` , modify `loda_path` as the model path and `write_path` as the path where the predicted results are saved in ``conf/predict.yaml`` , and then run `python predict.py`

.. code-block:: bash

    cd example/ner/few-shot

    python run.py
    
    python predict.py

standard_re
-----------
- The standard module is implemented by common deep learning models, including CNN, RNN, Capsule, GCN, Transformer and the pretrained model.
- Enter the ``DeepKE/example/re/standard`` folder. 
- The dataset and parameters can be customized in the ``data`` folder and ``conf`` folder respectively.
- **Train**: `python run.py`
- **Predict**: `python predict.py`

.. code-block:: bash

    cd example/re/standard

    python run.py

    python predict.py

few-shot_re
-----------
- This module is in the low-resouce scenario.
- Enter ``DeepKE/example/re/few-shot`` .
- **Train**: `python run.py`
  Start with the model trained last time: modify `train_from_saved_model` in ``conf/train.yaml`` as the path where the model trained last time was saved. 
  And the path saving logs generated in training can be customized by ``log_dir``.
- **Predict**: `python predict.py`

.. code-block:: bash

    cd example/re/few-shot

    python run.py
    
    python predict.py

document_re
-----------
- Download the model `train_distant.json` from [*Google Drive*](https://drive.google.com/drive/folders/1c5-0YwnoJx8NS6CV2f-NoTHR__BdkNqw) to ``data/``.
- Enter ``DeepKE/example/re/document`` .
- **Train**: `python run.py`
  Start with the model trained last time: modify `train_from_saved_model` in ``conf/train.yaml`` as the path where the model trained last time was saved. 
  And the path saving logs generated in training can be customized by ``log_dir``.
- **Predict**: `python predict.py`

.. code-block:: bash

    cd example/re/document

    python run.py

    python predict.py

standard_ae
-----------
- The standard module is implemented by common deep learning models, including CNN, RNN, Capsule, GCN, Transformer and the pretrained model.
- Enter the ``DeepKE/example/ae/standard`` folder. 
- The dataset and parameters can be customized in the ``data`` folder and ``conf`` folder respectively.
- **Train**: `python run.py`
- **Predict**: `python predict.py`

.. code-block:: bash

    cd example/ae/regular

    python run.py

    python predict.py


More details , you can refer to https://www.bilibili.com/video/BV1n44y1x7iW?spm_id_from=333.999.0.0 .