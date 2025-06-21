# mcp-tools

## 1. 简介

`DeepKE` 是一个开源的知识图谱抽取与构建工具，支持**cnSchema、低资源、长篇章、多模态**的知识抽取工具，提供标准化的 NLP 抽取任务，包括：

- ✅ 命名实体识别 (NER)
- ✅ 关系抽取 (RE)
- ✅ 属性抽取 (AE)
- ✅ 事件抽取 (EE)

`mcp-tools` 在 `DeepKE` 原有任务的基础上，提供标准化、可独立调用的 MCP 服务接口，为四个任务的**常规全监督standatd**、**预测**部分提供了可供大语言模型调用的 mcp 服务，在线服务可查看 [链接](https://modelscope.cn/mcp/servers/OpenKG/deepke-mcp-tools)使用。

## 2. 本地部署及配置

> 在使用前，请先完成 [DeepKE虚拟环境](https://github.com/zjunlp/DeepKE?tab=readme-ov-file#deepke)配置，准备好对应任务已经训练好的模型，并确认 DeepKE 中的 predict.py 可正常调用。具体教程可参考 [链接](https://github.com/zjunlp/DeepKE/blob/main/README_CNSCHEMA_CN.md)。
> 
> 如果您希望先简单验证 MCP 服务是否正常，可修改 `DeepKE/example/ner/standard/predict.py` 中的内容，例如
> ```
> print("hello world! this is only for test, not real code")
> ``` 
>，并且引导大语言模型来测试。

### 配置 `.env` 环境变量

将配置好的DeepKE虚拟环境`conda` 对应的包含 `PY` 的目录和 `DeepKE` 目录放入 `.env` 中，例如：

```
DEEPKE_PATH=".."
CONDA_PY="/home/user_name/anaconda3/envs/deepke/bin/"
CONDA_EE_PY="/home/user_name/anaconda3/envs/deepke-ee/bin/"
```

随后将您希望调用的模型名称`model_name`、`base_url`与`api_key`填入配置文件中。


### 配置mcp项目uv环境

```bash
pip install uv
cd mcp-tools
uv venv
source .venv/bin/activate
uv add "mcp[cli]" httpx openai pyyaml
```

### 测试运行
```bash
python run.py # 确保处于mcp-tools目录下
```

### 在 Cursor & Cline 中使用

修改 `.cursor/mcp.json` 或 `cline_mcp_settings.json` ，或者在其他支持 mcp 服务的应用中添加类似的配置。

```json
{
  "mcpServers": {
    "DeepKE": {
      "command": "uv",
      "args": [
        "--directory",
        "absolute/path/to/DeepKE/mcp-tools/tools",
        "run",
        "server.py"
      ]
    }
  }
}
```

## 3. 支持的能力

`mcp-tools` 当前提供以下任务接口：

| 任务类型 | 接口名         | 功能概述                         |
| -------- | -------------- | -------------------------------- |
| NER      | `deepke_ner()` | 常规全监督命名实体识别预测       |
| RE       | `deepke_re()`  | 常规全监督关系抽取预测           |
| AE       | `deepke_ae()`  | 常规全监督属性抽取预测           |
| EE       | `deepke_ee()`  | 常规全监督事件检测及论元提取预测 |

## 4. 注意事项

- 该 mcp 服务目前仍然处于早期阶段，相关能力正在完善中。
- 当前默认适配 Linux 路径，Windows 需修改对应路径。