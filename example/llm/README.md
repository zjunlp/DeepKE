<p align="left">
    <b> English | <a href="https://github.com/zjunlp/DeepKE/blob/main/example/llm/README_CN.md">简体中文</a> </b>
</p>

## Contents

- [Instruct Large Language Models using In-Context Learning](#instruct-large-language-models-using-in-context-learning)
  - [IE with Large Language Models](#ie-with-large-language-models)
  - [Data Augmentation with Large Language Models](#data-augmentation-with-large-language-models)
  - [CCKS2023 Instruction-based Knowledge Graph Construction with Large Language Models](#ccks2023-instruction-based-knowledge-graph-construction-with-large-language-models)
- [CodeKGC-Code Language Models for Knowledge Graph Construction](#codekgc-code-language-models-for-knowledge-graph-construction)


# Instruct Large Language Models using In-Context Learning

## IE with Large Language Models

## Data Augmentation with Large Language Models

To compensate for the lack of labeled data in few-shot scenarios for relation extraction, we have designed prompts with data style descriptions to guide large language models to automatically generate more labeled data based on existing few-shot data.

## CCKS2023 Instruction-based Knowledge Graph Construction with Large Language Models


# CodeKGC-Code Language Models for Knowledge Graph Construction

To better address Relational Triple Extraction (rte) task in Knowledge Graph Construction, we have designed code-style prompts to model the structure of  Relational Triple, and used Code-LLMs to generate more accurate predictions. The key step of code-style prompt construction is to transform (text, output triples) pairs into semantically equivalent program language written in Python.

<div align=center>
<img src="./CodeKGC/codekgc_figure.png" width="85%" height="75%" />
</div>



