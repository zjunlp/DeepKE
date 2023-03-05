1. Mainfest:
- 53_schema.jsonl: SPO关系约束表
- CMeIE_train.jsonl: 训练集
- CMeIE_dev.jsonl: 验证集
- CMeIE_test.jsonl: 测试集,选手提交的时候需要为每条记录填充"spo_list"字段，类型为列表。每个识别出来的关系必须包含"subject", "predicate", "object"3个字段，且"object"是一个字典（和训练数据保持一致）: {"@value": "some string"}。
- example_gold.jsonl: 标准答案示例
- example_pred.jsonl: 提交结果示例
- README.txt: 说明文件

2. 评估指标以严格Micro-F1值为准

3. 该任务提交的文件名为：CMeIE_test.jsonl
