import os
import subprocess
import yaml
import json
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from typing import Literal, Optional
from convert_to_tsv import input_to_raw_and_tsv

load_dotenv()

mcp = FastMCP("DeepKE")

TaskType = Literal["standard", "few-shot", "multimodal", "documental"]  # 添加documental类型

@mcp.tool()
def deepke_ner(task: TaskType, txt: str) -> str:
    """
    使用 deepke 运行 NER 预测任务
    
    参数:
    - task: 任务类型（目前只支持standard）
    - txt: 用户的预测文本
    
    返回:
    - 命令输出或错误信息
    """
    if task != "standard":
        return "[NER任务目前只支持standard模式]"
    
    base_path = os.path.expanduser(os.path.join(
        os.getenv("DEEPKE_PATH"), "example/ner/standard"
    ))
    config_path = os.path.join(base_path, "conf/predict.yaml")
    
    try:
        # 修改配置文件
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config["text"] = txt
        
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True)
        
        # 执行预测
        command = f"cd {base_path} && {os.getenv('CONDA_PY')}python predict.py"
        result = subprocess.run(
            ["bash", "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return result.stdout
        
    except Exception as e:
        return f"[运行失败] {str(e)}"

@mcp.tool()
def deepke_re(
    task: TaskType,
    txt: str,
    head: str,
    head_type: str,
    tail: str,
    tail_type: str
) -> str:
    """
    使用 deepke 运行关系抽取(RE)预测任务
    
    参数:
    - task: 任务类型（目前只支持standard）
    - txt: 包含实体的句子
    - head: 句中需要预测关系的头实体
    - head_type: 头实体类型
    - tail: 句中需要预测关系的尾实体
    - tail_type: 尾实体类型
    
    返回:
    - 命令输出或错误信息
    """
    if task != "standard":
        return "[RE任务目前只支持standard模式]"
    
    base_path = os.path.expanduser(os.path.join(
        os.getenv("DEEPKE_PATH"), "example/re/standard"
    ))
    
    try:
        # 构建输入文本格式：句子 + 实体信息
        input_text = f"n\n{txt}\n{head}\n{head_type}\n{tail}\n{tail_type}\n"
        
        # 执行预测（通过标准输入传递）
        command = f"cd {base_path} && {os.getenv('CONDA_PY')}python predict.py"
        process = subprocess.Popen(
            ["bash", "-c", command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=input_text)
        
        return stdout if process.returncode == 0 else f"[运行失败]\n{stderr}"
        
    except Exception as e:
        return f"[运行异常] {str(e)}"
    
@mcp.tool()
def deepke_ae(
    txt: str,
    entity: str,
    attribute_value: str,
    task: TaskType = 'standard'
) -> str:
    """
    使用 deepke 运行属性抽取(AE)预测任务
    
    参数:
    - txt: 需要做属性抽取的句子
    - entity: 句中需要预测的实体
    - attribute_value: 句中需要预测的属性值
    
    返回:
    - 命令输出或错误信息
    """
    
    base_path = os.path.expanduser(os.path.join(
        os.getenv("DEEPKE_PATH"), "example/ae/standard"
    ))
    
    try:
        # 构建输入文本格式
        input_text = f"n\n{txt}\n{entity}\n{attribute_value}\n"
        
        # 执行预测（通过标准输入传递）
        command = f"cd {base_path} && {os.getenv('CONDA_PY')}python predict.py"
        process = subprocess.Popen(
            ["bash", "-c", command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=input_text)
        
        return stdout if process.returncode == 0 else f"[运行失败]\n{stderr}"
        
    except Exception as e:
        return f"[运行异常] {str(e)}"
    
@mcp.tool()
def deepke_ee(txt: str) -> str:
    """
    使用 deepke 运行事件抽取 EE 预测任务
    
    参数:
    - txt: 用户的预测文本
    
    返回:
    - 命令输出或错误信息
    """
    
    base_path = os.path.expanduser(os.path.join(
        os.getenv("DEEPKE_PATH"), "example/ee/standard"
    ))
    trigger_eval_path = os.path.join(base_path, "exp/DuEE/trigger/bert-base-chinese/eval_pred.json")
    trigger_result_path = os.path.join(base_path, "exp/DuEE/trigger/bert-base-chinese/eval_results.txt")
    role_eval_path = os.path.join(base_path, "exp/DuEE/role/bert-base-chinese/eval_pred.json")
    role_result_path = os.path.join(base_path, "exp/DuEE/role/bert-base-chinese/eval_results.txt")
    train_config_path = os.path.join(base_path, "conf/train.yaml")

    # 转换输入为测试文件
    input_to_raw_and_tsv(txt, os.path.join(base_path, "data/DuEE/raw/duee_dev.json"), 
                                os.path.join(base_path, "data/DuEE/role/dev.tsv"), 
                                os.path.join(base_path, "data/DuEE/trigger/dev.tsv"))
    
    try:
        # 修改配置文件
        with open(train_config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config["task_name"] = "trigger"
        
        with open(train_config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True)
        
        # 执行trigger预测
        command = f"cd {base_path} && {os.getenv('CONDA_EE_PY')}python run.py"
        subprocess.run(
            ["bash", "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        # 修改配置文件
        with open(train_config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config["task_name"] = "role"
        
        with open(train_config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True)

        # 训练一个事件角色抽取的模型
        command = f"cd {base_path} && {os.getenv('CONDA_EE_PY')}python run.py"
        subprocess.run(
            ["bash", "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        # 执行论元预测
        command = f"cd {base_path} && {os.getenv('CONDA_EE_PY')}python predict.py"
        subprocess.run(
            ["bash", "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        # 读取结果文件
        with open(trigger_eval_path, "r", encoding="utf-8") as f:
            trigger_data = f.read()
        with open(trigger_result_path, "r", encoding="utf-8") as f:
            trigger_result = f.read()
        with open(role_eval_path, "r", encoding="utf-8") as f:
            role_data = f.read()
        with open(role_result_path, "r", encoding="utf-8") as f:
            role_result = f.read()

        result = (
            "请核对结果，根据分隔一一确定预测内对应原句位置的内容，B，I，O分别代表起始，中间和无关字。\n"
            "[触发词预测结果]\n"
            f"{trigger_result.strip()}\n\n"
            "[触发词预测内容]\n"
            f"{trigger_data.strip()}\n\n"
            "[论元预测结果]\n"
            f"{role_result.strip()}\n\n"
            "[论元预测内容]\n"
            f"{role_data.strip()}"
        )

        return result

    except Exception as e:
        return f"[运行失败] {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")