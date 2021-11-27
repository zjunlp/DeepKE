FROM ubuntu:18.04
LABEL maintainer="ZJUNLP"
LABEL repository="DeepKE"
ENV PYTHON_VERSION=3.8

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         wget \
         git \
         curl \
         ca-certificates
                   
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-4.7.12-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b && \
    rm ~/miniconda.sh

ENV PATH=/root/miniconda3/bin:$PATH

RUN conda create -y --name deepke python=$PYTHON_VERSION

# SHELL ["/root/miniconda3/bin/conda", "run", "-n", "deepke", "/bin/bash", "-c"]
RUN conda init bash 

RUN cd ~ && \
    git clone https://github.com/zjunlp/DeepKE.git 
    