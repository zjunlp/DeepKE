from setuptools import setup, find_packages

setup(
    name="cpm_live",
    version="0.1.0",
    author="OpenBMB",
    author_email="openbmb@gmail.com",
    description="Toolkit for CPM-Live",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch>=1.10",
        "bmtrain>=0.2.0",
        "jieba",
        "tqdm",
        "tensorboard",
        "numpy>=1.21.0",
        "opendelta",
    ],
    package_data={"cpm_live": ["vocabs/*.txt"]},
)
