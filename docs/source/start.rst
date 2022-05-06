Start
=====



Model Framework
---------------

.. image:: ./_static/architectures.png

DeepKE contains three modules for named entity recognition, relation extraction and attribute extraction, the three tasks respectively.

Each module has its own submodules. For example, there are standard, few-shot, document-level and multimodal submodules in the relation extraction modular.

Each submodule compose of three parts: a collection of tools, which can function as tokenizer, dataloader, preprocessor and the like, a encoder and a part for training and prediction.

Dataset
-------

We use the following datasets in our experiments:

+--------------------------+-----------+------------------+----------+------------+
| Task                     | Settings  | Corpus           | Language |  Model     |
+==========================+===========+==================+==========+============+
|                          |           | CoNLL-2003       | English  |            |
|                          | Standard  +------------------+----------+  BERT      |
|                          |           | People's Daily   | Chinese  |            |
|                          +-----------+------------------+----------+------------+
|                          |           | CoNLL-2003       |          |            |
|                          |           +------------------+          |            |
| Name Entity Recognition  |           | MIT Movie        |          |            |
|                          | Few-shot  +------------------+ English  | LightNER   |
|                          |           | MIT Restaurant   |          |            |
|                          |           +------------------+          |            |
|                          |           | ATIS             |          |            |  
|                          +-----------+------------------+----------+------------+
|                          |           | Twitter15        |          |            |
|                          | Multimodal+------------------+ English  | IFAformer  |
|                          |           | Twitter17        |          |            |
+--------------------------+-----------+------------------+----------+------------+
|                          |           |                  |          | CNN        |
|                          |           |                  |          +------------+
|                          |           |                  |          | RNN        |
|                          |           |                  |          +------------+
|                          |           |                  |          | Capsule    |
|                          | Standard  | DuIE             | Chinese  +------------+
|                          |           |                  |          | GCN        |
|                          |           |                  |          +------------+
|                          |           |                  |          | Transformer|
|                          |           |                  |          +------------+
|                          |           |                  |          | BERT       |
|                          +-----------+------------------+----------+------------+
| Relation Extraction      |           | SEMEVAL(8-shot)  |          |            |
|                          |           +------------------+          |            |
|                          |           | SEMEVAL(16-shot) |          |            |
|                          | Few-shot  +------------------+ English  | KnowPrompt |
|                          |           | SEMEVAL(32-shot) |          |            |
|                          |           +------------------+          |            |
|                          |           | SEMEVAL(Full)    |          |            |
|                          +-----------+------------------+----------+------------+
|                          |           | DocRED           |          |            |
|                          |           +------------------+          |            |
|                          | Document  | CDR              | English  | DocuNet    |
|                          |           +------------------+          |            |
|                          |           | GDA              |          |            |
|                          +-----------+------------------+----------+------------+
|                          | Multimodal| MNRE             | English  | IFAformer  |
+--------------------------+-----------+------------------+----------+------------+
|                          |           |                  |          | CNN        |
|                          |           |                  |          +------------+
|                          |           |                  |          | RNN        |
|                          |           |                  |          +------------+
|                          |           |Triplet Extraction|          | Capsule    |
| Attribute Extraction     | Standard  |Dataset           | Chinese  +------------+
|                          |           |                  |          | GCN        |
|                          |           |                  |          +------------+
|                          |           |                  |          | Transformer|
|                          |           |                  |          +------------+
|                          |           |                  |          | BERT       |
+--------------------------+-----------+------------------+----------+------------+


Get Start
---------

If you want to use our code , you can do as follow:

.. code-block:: python

     git clone https://github.com/zjunlp/DeepKE.git
     cd DeepKE



