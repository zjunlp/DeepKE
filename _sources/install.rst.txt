Install
=======

Create environment
------------------

Create a virtual environment directly (recommend anaconda)

.. code-block:: bash

    conda create -n deepke python=3.8
    conda activate deepke

We also provide dockerfile to create docker image.

.. code-block:: bash

    cd docker
    docker build -t deepke .
    conda activate deepke

Install by pypi
---------------

If use deepke directly

.. code-block:: python

    pip install deepke


Install by setup.py
-------------------

If modify source codes before usage

.. code-block:: python

    python setup.py install