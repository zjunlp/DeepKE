from setuptools import setup, find_packages
setup(
    name='deepke',  # 打包后的包文件名
    version='0.2.25',    #版本号
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
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
)
