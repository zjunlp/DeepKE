from setuptools import setup, find_packages


# with open("requirements.txt") as requirements_file:
#     requirements = requirements_file.read().splitlines()
    
setup(
    name='deepke',  # 打包后的包文件名
    version='2.2.7',    #版本号
    keywords=["pip", "RE","NER","AE"],    # 关键字
    description='DeepKE is a knowledge extraction toolkit for knowledge graph construction supporting low-resource, document-level and multimodal scenarios for entity, relation and attribute extraction.',  # 说明
    license="MIT",  # 许可
    url='https://github.com/zjunlp/deepke',
    author='ZJUNLP',
    author_email='zhangningyu@zju.edu.cn',
    include_package_data=True,
    platforms="any",
    package_dir={"": "src"},
    packages=find_packages("src"),
    # install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
)
