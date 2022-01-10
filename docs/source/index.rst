
DeepKE Documentation
====================

Introduction
------------


.. image:: ./_static/logo.png

DeepKE is a knowledge extraction toolkit supporting low-resource and document-level scenarios. It provides three functions based PyTorch, including Named Entity Recognition, Relation Extraciton and Attribute Extraction.


.. image:: ./_static/demo.gif

Support Weight & Biases
-----------------------

.. image:: ./_static/wandb.png

To achieve automatic hyper-parameters fine-tuning, DeepKE adopts Weight & Biases, a machine learning toolkit for developers to build better models faster.
With this toolkit, DeepKE can visualize results and tune hyper-parameters better automatically.
The example running files for all functions in the repository support the toolkit and researchers are able to modify the metrics and hyper-parameter configuration as needed. 
The detailed usage of this toolkit refers to the official document

Support Notebook Tutorials
--------------------------

We provide Google Colab tutorials and jupyter notebooks in the github repository as example implementation of every functions in different scenarios. 
These tutorials can be run directly and lead developers and researchers to have a whole picture of DeepKE’s application methods.

You can go colab directly: https://colab.research.google.com/drive/1cM-zbLhEHkje54P0IZENrfe4HaXwZxZc?usp=sharing

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Getting Started

   start
   install
   example
   faq



.. toctree::
   :glob:
   :maxdepth: 3
   :caption: Package 

   deepke

