from setuptools import setup, find_packages

setup(
    name='deepke',  # 打包后的包文件名
    version='2.0.1',    #版本号
    keywords=["pip", "RE","NER","AE"],    # 关键字
    description='DeepKE 是基于 Pytorch 的深度学习中文关系抽取处理套件。',  # 说明
    long_description="client",  #详细说明
    license="MIT",  # 许可
    url='https://github.com/zjunlp/deepke',
    author='ZJUNLP',
    author_email='zhangningyu@zju.edu.cn',
    include_package_data=True,
    platforms="any",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        'torch>=1.5,<=1.10',
        'hydra-core==1.0.6',
        'tensorboard==2.4.1',
        'matplotlib==3.4.1',
        'transformers==3.4.0',
        'jieba==0.42.1',
        'scikit-learn==0.24.1',
        'pytorch-transformers==1.2.0',
        'seqeval==1.2.2',
        'tqdm==4.60.0',
        'opt-einsum==3.3.0',
        'wandb==0.12.7',
        "ujson",
        "huggingface_hub==0.0.8"
    ], 
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
)
