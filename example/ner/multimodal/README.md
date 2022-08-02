# Easy Start

<p align="left">
    <b> English | <a href="https://github.com/zjunlp/DeepKE/blob/main/example/ner/multimodal/README_CN.md">简体中文</a> </b>
</p>

## Model

**IFAformer** is a novel dual Multimodal Transformer model with implicit feature alignment, which utilizes the Transformer structure uniformly in the visual and textual without explicitly designing modal alignment structure.

<div align=center>
<img src="mner_model.png" width="75%" height="75%"/>
</div>

## Experiment

The overall experimental results on IFAformer for Multi-Modal NER task can be seen as follows:

<table>
	<tr>
		<th></th>
		<th>Methods</th>
		<th>Precision</th>
		<th>Recall</th>
		<th>F1</th>
	</tr>
	<tr>
		<td rowspan="3">text</td>
		<td>CNN-BiLSTM-(CRF)</td>
		<td>80.00</td>
		<td>78.76</td>
		<td>79.37</td>
	</tr>
	<tr>
		<td>BERT-(CRF)</td>
		<td>83.32</td>
		<td>83.57</td>
		<td>83.44</td>
	</tr>
	<tr>
		<td>MTB</td>
		<td>83.88</td>
		<td>83.22</td>
		<td>83.55</td>
	</tr>
	<tr>
		<td rowspan="5">text+image</td>
		<td>AdapCoAtt-BERT-(CRF)</td>
		<td>85.13</td>
		<td>83.20</td>
		<td>84.10</td>
	</tr>
	<tr>
		<td>VisualBERT_base</td>
		<td>84.06</td>
		<td>85.39</td>
		<td>84.72</td>
	</tr>
	<tr>
		<td>ViLBERT_base</td>
		<td>84.62</td>
		<td>85.47</td>
		<td>85.04</td>
	</tr>
	<tr>
		<td>UMT</td>
		<td>85.28</td>
		<td>85.34</td>
		<td>85.31</td>
	</tr>
	<tr>
		<td><b>IFAformer</b></td>
		<td><b>86.88</b></td>
		<td><b>87.91</b></td>
		<td><b>87.39</b></td>
	</tr>
</table>

## Requirements

> python == 3.8

- torch == 1.5
- transformers == 3.4.0
- hydra-core == 1.0.6
- deepke

<font color='red'> Attention! </font>
Here `transformers==3.4.0` is the environmental requirement of the whole `DeepKE`. But to load the `openai/clip-vit-base-patch32` model used in multimodal parts, `transformers==4.11.3` is needed indeed. So you are recommended to download the [pretrained model](https://huggingface.co/openai/clip-vit-base-patch32) on huggingface and use the local path to load the model.

## Download Code

```bash
git clone https://github.com/zjunlp/DeepKE.git
cd DeepKE/example/ner/multimodal
```

## Install with Pip

- Create and enter the python virtual environment.
- Install dependencies: `pip install -r requirements.txt`.

## Train and Predict

- Dataset - Twitter2015 & Twitter2017
  
  - Download the dataset to this directory.
    
    The text data follows the conll format.
    The acquisition of Twitter15 and Twitter17 data refer to the code from [UMT](https://github.com/jefferyYu/UMT/), many thanks.
    
    You can download the Twitter2015 and Twitter2017 dataset with detected visual objects using folloing command:
    
    ```bash
    wget 120.27.214.45/Data/ner/multimodal/data.tar.gz
    tar -xzvf data.tar.gz
    ```
  - The twitter15 dataset with detected visual objects is stored in `data`:
    
    - `twitter15_detect`：Detected objects using RCNN
    - `twitter2015_aux_images`：Detected objects using visual grouding
    - `twitter2015_images`： Original images
    - `train.txt`: Train set
    - `...`
- Training
  
  - Parameters, model paths and configuration for training are in the `conf` folder and users can modify them before training.
  - Download the [PLM](https://huggingface.co/openai/clip-vit-base-patch32/tree/main) and set `vit_name` in `train.yaml` and `predict.yaml` as the directory of the PLM.
  - Run
    
    ```bash
    python run.py
    ```
  - The trained model is stored in the `checkpoint` directory by default and you can change it by modifying "save_path" in `train.yaml`.
  - Start to train from last-trained model `<br>`
    
    modify `load_path` in `train.yaml` as the path of the last-trained model
  - Logs for training are stored in the current directory by default and the path can be configured by modifying `log_dir` in `.yaml`
- Prediction
  
  Modify "load_path" in `predict.yaml` to the trained model path. **In addition, we provide [the model trained on Twitter2017 dataset](https://drive.google.com/drive/folders/1ZGbX9IiNU3cLZtt4U8oc45zt0BHyElAQ?usp=sharing) for users to predict directly.**
  
  ```bash
  python predict.py
  ```

