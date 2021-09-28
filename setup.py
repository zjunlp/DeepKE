from setuptools import setup, find_packages
setup(
    name='deepke',  # 打包后的包文件名
    version='0.2.60',    #版本号
    keywords=["pip", "RE","NER","AE"],    # 关键字
    description='DeepKE 是基于 Pytorch 的深度学习中文关系抽取处理套件。',  # 说明
    long_description="client",  #详细说明
    license="Apache-2.0 License",  # 许可
    url='https://github.com/zjunlp/deepke',
    author='ZJUNLP',
    author_email='xx2020@zju.edu.cn',
    include_package_data=True,
    platforms="any",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        'torch>=1.2',
        'hydra-core==1.0.6',
        'tensorboard>=2.0',
        'matplotlib>=3.1',
        'transformers>=2.0',
        'jieba>=0.39',
        'scikit-learn>=0.22',
        'pytorch-transformers>=1.2.0',
        'seqeval>=0.0.5',
        'tqdm>=4.31.1',
        'nltk>=3.4.5'
    ], 
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
)
