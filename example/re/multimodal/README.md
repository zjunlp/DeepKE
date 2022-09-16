# Easy Start

<p align="left">
    <b> English | <a href="https://github.com/zjunlp/DeepKE/blob/main/example/re/multimodal/README_CN.md">简体中文</a> </b>
</p>

## Model

**IFAformer** is a novel dual Multimodal Transformer model with implicit feature alignment, which utilizes the Transformer structure uniformly in the visual and textual without explicitly designing modal alignment structure.

<div align=center>
<img src="mre_model.png" width="75%" height="75%"/>
</div>

## Experiment

The overall experimental results on IFAformer for Multi-Modal RE task can be seen as follows:

<table>
	<tr>
		<th></th>
		<th>Methods</th>
		<th>Acc</th>
		<th>Precision</th>
		<th>Recall</th>
		<th>F1</th>
	</tr>
	<tr>
		<td rowspan="3">text</td>
		<td>PCNN*</td>
		<td>73.36</td>
		<td>69.14</td>
		<td>43.75</td>
		<td>53.59</td>
	</tr>
	<tr>
		<td>BERT*</td>
		<td>71.13</td>
		<td>58.51</td>
		<td>60.16</td>
		<td>59.32</td>
	</tr>
	<tr>
		<td>MTB*</td>
		<td>75.34</td>
		<td>63.28</td>
		<td>65.16</td>
		<td>64.20</td>
	</tr>
	<tr>
		<td rowspan="4">text+image</td>
	</tr>
	<tr>
		<td>BERT+SG+Att</td>
		<td>74.59</td>
		<td>60.97</td>
		<td>66.56</td>
		<td>63.64</td>
	</tr>
	<tr>
		<td>ViLBERT</td>
		<td>74.89</td>
		<td>64.50</td>
		<td>61.86</td>
		<td>63.61</td>
	</tr>
	<tr>
		<td>MEGA</td>
		<td>76.15</td>
		<td>64.51</td>
		<td>68.44</td>
		<td>66.41</td>
	</tr>
	<tr>
		<td rowspan="4">Ours</td>
	</tr>
	<tr>
		<td>Vanilla IFAformer</td>
		<td>87.75</td>
		<td>69.90</td>
		<td>68.11</td>
		<td>68.99</td>
	</tr>
	<tr>
		<td>&emsp;w/o Text Attn.</td>
		<td>76.21</td>
		<td>66.95</td>
		<td>61.72</td>
		<td>64.23</td>
	</tr>
	<tr>
		<td>&emsp;w/ Visual Objects</td>
		<td><b>92.38</b></td>
		<td><b>82.59</b></td>
		<td><b>80.78</b></td>
		<td><b>81.67</b></td>
	</tr>
</table>

## Requirements

> python == 3.8

- torch == 1.5
- transformers == 3.4.0
- hydra-core == 1.0.6
- deepke

<font color='red'> Attention! </font>
Here `transformers == 3.4.0` is the environmental requirement of the whole `DeepKE`. But to load the `openai/clip-vit-base-patch32` model used in multimodal parts, `transformers == 4.11.3` is needed indeed. So you are recommended to download the [pretrained model](https://huggingface.co/openai/clip-vit-base-patch32) on huggingface and use the local path to load the model.

## Download Code

```bash
git clone https://github.com/zjunlp/DeepKE.git
cd DeepKE/example/re/multimodal
```

## Install with Pip

- Create and enter the python virtual environment.
- Install dependencies: `pip install -r requirements.txt`.

## Train and Predict

- Dataset
  
  - Download the dataset to this directory.
    
    The MNRE dataset comes from [https://github.com/thecharm/Mega](https://github.com/thecharm/Mega), many thanks.
    
    You can download the MNRE dataset with detected visual objects using folloing command:
    
    ```bash
    wget 120.27.214.45/Data/re/multimodal/data.tar.gz
    tar -xzvf data.tar.gz
    ```
  - The dataset [MNRE](https://github.com/thecharm/Mega) with detected visual objects is stored in `data`:
    
    - `img_detect`：Detected objects using RCNN
    - `img_vg`：Detected objects using visual grounding
    - `img_org`： Original images
    - `txt`: Text set
    - `vg_data`：Bounding image and `img_vg`
    - `ours_rel2id.json` Relation set
  - We use RCNN detected objects and visual grounding objects as visual local information, where RCNN via [faster_rcnn](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py) and visual grounding via [onestage_grounding](https://github.com/zyang-ur/onestage_grounding).
- Training
  
  - Parameters, model paths and configuration for training are in the `conf` folder and users can modify them before training.
  - Training on MNRE
    
    ```bash
    python run.py
    ```
  - The trained model is stored in the `checkpoint` directory by default and you can change it by modifying "save_path" in `train.yaml`.
  - Start to train from last-trained model
    
    modify `load_path` in `train.yaml` as the path of the last-trained model
  - Logs for training are stored in the current directory by default and the path can be configured by modifying `log_dir` in `.yaml`
- Prediction
  
  Modify "load_path" in `predict.yaml` to the trained model path.  **In addition, we provide [the model trained on MNRE dataset](https://drive.google.com/drive/folders/11T0t1NHSMq5GzORBKv2Rjm2Bbq_RNLrc?usp=sharing) for users to predict directly.**
  
  ```bash
  python predict.py
  ```

