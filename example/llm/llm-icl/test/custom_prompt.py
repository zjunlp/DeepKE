"""
custom_prompt.py

        prompt:分中英文两情况（2）。分re、da、其他三情况（3）。分是否incontext两情况（2）。用build_prompt()加工。最终是instruction+prompt。
        response:
        output:

        task:必须为["ner", "re", "ee", "rte", "da"]

        instruction:如果为空，则用函数get_default_instruction()造；如果不为空，则添加换行符
        language:必须为["en", "ch"]
        domain:

        head_entity:
        head_type:
        tail_entity:
        tail_type:

        (bool) in_context:
        (dict) examples:如果in_context为True，则必须要有examples，且examples需要用get_incontext_examples()加工。
        (list) labels:如果是da任务，则必须要有labels
"""

from typing import Union, Dict, List


NER_EN_DOMAIN_LABELS = "You are a highly intelligent and accurate {0} domain Named-entity recognition(NER) system. You take Passage as input and your task is to recognize and extract specific types of {0} domain named entities in that given passage and classify into a set of following predefined entity types:\n{1}.\n"
NER_EN_DOMAIN = "You are a highly intelligent and accurate {0} domain Named-entity recognition(NER) system. You take Passage as input and your task is to recognize and extract specific types of {0} domain named entities in that given passage and classify into a set of entity types.\n"
NER_EN_LABELS = "You are a highly intelligent and accurate Named-entity recognition(NER) system. You take Passage as input and your task is to recognize and extract specific types of named entities in that given passage and classify into a set of following predefined entity types:\n{0}.\n"
NER_EN = "You are a highly intelligent and accurate relation extraction(RE) system. Given a context, a pair of head and tail entities in the context, your task is to extract the specific type of relationship between the head and tail entities.\n"
NER_EN_OUTPUT = "Your output format is only [{'E': type of entity from predefined entity types, 'W': entity in the input text},...] form, no other form.\n\n"

RE_EN_DOMAIN_LABELS = "You are a highly intelligent and accurate {0} domain relation extraction(RE) system. Given a context, a pair of head and tail entities in the context, your task is to extract the specific type of {0} domain relationship between the head and tail entities from candidate relations:\n{1}.\n"
RE_EN_DOMAIN = "You are a highly intelligent and accurate {0} domain relation extraction(RE) system. Given a context, a pair of head and tail entities in the context, your task is to extract the specific type of {0} domain relationship between the head and tail entities.\n"
RE_EN_LABELS = "You are a highly intelligent and accurate relation extraction(RE) system. Given a context, a pair of head and tail entities in the context, your task is to extract the specific type of relationship between the head and tail entities from candidate relations:\n{0}.\n"
RE_EN = "You are a highly intelligent and accurate relation extraction(RE) system. Given a context, a pair of head and tail entities in the context, your task is to extract the specific type of relationship between the head and tail entities.\n"
RE_EN_OUTPUT = "Your output is only the relation type, no other words.\n\n"

EE_EN_DOMAIN = "You are a highly intelligent and accurate {0} domain event extraction model. You take Passage as input and convert it into {0} domain events arguments.You can identify all events of target types mentioned in the sentence, and extract corresponding event arguments playing target roles.\n"
EE_EN = "You are a highly intelligent and accurate event extraction model. You take Passage as input and convert it into events arguments. You can identify all events of target types mentioned in the sentence, and extract corresponding event arguments playing target roles.\n"
EE_EN_OUTPUT = "Your output format is only [{event_type, arguments: [{role , argument}, ...]}, ...], nothing else.\n\n"

RTE_EN_DOMAIN = "You are a highly intelligent and accurate {0} domain Resource Description Framework (RDF) data model. You take Passage as input and convert it into {0} domain RDF triples. A triple is a set of three entities that codifies a statement about semantic data in the form of subject-predicate-object expressions.\n"
RTE_EN = "You are a highly intelligent and accurate Resource Description Framework (RDF) data model. You take Passage as input and convert it into RDF triples. A triple is a set of three entities that codifies a statement about semantic data in the form of subject-predicate-object expressions.\n"
RTE_EN_OUTPUT = "Your output format is only [[ subject, predicate, object ], ...], nothing else.\n\n"

DA_EN_LABELS = "One sample in relation extraction datasets consists of a relation, a context, a pair of head and tail entities in the context and their entity types. The head entity has the relation with the tail entity and entities are pre-categorized as the following types: {0}.\n\n"


NER_CH_DOMAIN_LABELS = "您是一个高度智能和精确的{0}域命名实体识别（NER）系统。您将文本作为输入，您的任务是识别和提取给定段落中的特定类型的{0}域名实体，并将其分类为一组预定义的实体类型：\n{1}.\n"
NER_CH_DOMAIN = "您是一个高度智能和精确的{0}域命名实体识别（NER）系统。您将文本作为输入，您的任务是识别和提取给定文章中特定类型的{0}域命名实体，并将其分类为一组实体类型。\n"
NER_CH_LABELS = "您是一个高度智能和精确的命名实体识别（NER）系统。您将文本作为输入，您的任务是识别和提取给定段落中的特定类型的命名实体，并将其分类为一组预定义的实体类型：\n{0}.\n"
NER_CH = "您是一个高度智能和精确的命名实体识别（NER）系统。您将文本作为输入，您的任务是识别和提取给定文章中特定类型的命名实体，并将其分类为一组实体类型。\n"
NER_CH_OUTPUT = "您输出的格式需要为[{'E': 预先定义的实体类型, 'W': 输入文本中的实体},...]，没有其他格式要求。\n\n"

RE_CH_DOMAIN_LABELS = "您是一个高度智能和精确的{0}域关系抽取（RE）系统。给定上下文以及上下文中包含的一对头实体和尾实体，您的任务是提取给定头实体和尾实体间特定类型的{0}域关系，候选的关系类型如下：\n{1}.\n"
RE_CH_DOMAIN = (
    "您是一个高度智能和精确的{0}域关系抽取（RE）系统。给定上下文以及上下文中包含的一对头实体和尾实体，您的任务是提取给定头实体和尾实体间特定类型的{0}域关系。\n"
)
RE_CH_LABELS = "您是一个高度智能和精确的关系抽取（RE）系统。给定上下文以及上下文中包含的一对头实体和尾实体，您的任务是提取给定头实体和尾实体间特定类型的关系，候选的关系类型如下：\n{0}.\n"
RE_CH = "您是一个高度智能和精确的关系抽取（RE）系统。给定上下文以及上下文中包含的一对头实体和尾实体，您的任务是提取给定头实体和尾实体间特定类型的关系。\n"
RE_CH_OUTPUT = "您只需要输出关系的类型即可，不需要其他的文字输出。\n\n"

EE_CH_DOMAIN = "您是一个高度智能和精确的{0}域事件提取模型。您将文本作为输入并将其转换为{0}域事件参数。您可以识别句子中提到的所有目标类型的事件，并提取扮演目标角色的相应事件参数。\n"
EE_CH = "您是一个高度智能和精确的事件提取模型。您将文本作为输入并将其转换为事件参数。您可以识别句子中提到的所有目标类型的事件，并提取扮演目标角色的相应事件参数。\n"
EE_CH_OUTPUT = (
    "您的输出格式为 [{event_type, arguments: [{role , argument}, ...]}, ...]，没有其他要求。"
)

RTE_CH_DOMAIN = "您是一个高度智能和精确的{0}域资源描述框架（RDF）数据模型。您将文本作为输入，并将其转换为{0}域RDF三元组。三元组是由三个实体组成的集合，以主语-谓语-宾语表达式的形式对语义数据进行编码。\n"
RTE_CH = "您是一个高度智能和精确的资源描述框架（RDF）数据模型。您将文本作为输入，并将其转换为RDF三元组。三元组是由三个实体组成的集合，以主语-谓语-宾语表达式的形式对语义数据进行编码。\n"
RTE_CH_OUTPUT = "您输出的格式需要为[[ 主语, 谓语, 宾语 ], ...]，没有其他格式要求。\n\n"

DA_CH_LABELS = (
    "关系提取数据集中的一个样本由关系、文本、文本中的一对头实体和尾实体及它们的实体类型组成。头实体与尾实体间存在关系，头尾实体被预先分类为以下类型：{0}.\n\n"
)


class CustomPrompt:
    """
    Custom class for information extraction prompts.
    """
    def __init__(self, task: str = 'ner', language = 'en'):
        if task not in ["ner", "re", "ee", "rte", "da"]:
            raise ValueError(
                "The task name should be one of ner, re, ee, rte for Information Extraction or da for Data Augmentation."
            )
        if language not in ["en", "ch"]:
            raise ValueError(
                "Now we only support language of 'en' (English) and 'ch' (Chinese)."
            )
        self.task = task
        self.language = language

    def build_prompt(
            self,
            prompt : str,
            instruction: str = None,
            in_context: bool = None,
            examples: Dict = None,
            labels: List = None,
            head_entity: str = None,
            head_type: str = None,
            tail_entity: str = None,
            tail_type: str = None,
            domain: str = None
    ):
        # 检错
        if in_context and examples is None:
            raise ValueError("Please provide some examples if in-context=True.")
        if self.task == "da" and labels is None:
            raise ValueError(
                "Please provide some pre-categorized entity types if the task is Data Augmentation(da)."
            )
        # 预处理
        if instruction is None:
            instruction = self._get_default_instruction(domain, labels)
        else:
            instruction += '\n'
        # 提示词设计
        if self.language == 'en':
            if self.task == "re":
                prompt += f"Context: {prompt}\nThe relation between ({head_type}) '{head_entity}' and ({tail_type}) '{tail_entity}' in the context is"
            elif self.task == "da":
                prompt += f"Generate more samples for the relation '{prompt}'.\n"
            else:
                prompt = f"Input: {prompt}\nOutput: "
        elif self.language == 'ch':
            if self.task == "re":
                prompt += f"上下文：{prompt}\n上下文中头实体（{head_type}）‘{head_entity}’和尾实体（{tail_type}）‘{tail_entity}’之间的关系类型是"
            elif self.task == "da":
                prompt += f"请为关系‘{prompt}’生成更多的样例数据。\n"
            else:
                prompt = f"输入：{prompt}\n输出："
        # 上下文设计
        if in_context:
            examples = self._get_incontext_examples(examples)
            final_prompt = self.build_prompt_by_shots(
                prompt=prompt, in_context_examples=examples, n_shots=len(examples)
            )
        else:
            final_prompt = prompt
        # 返回
        final_prompt = instruction + final_prompt
        print(final_prompt)
        return  final_prompt

    def _get_default_instruction(self, domain, labels):
        """
        _get_default_instruction 方法生成每种任务类型的默认指令内容，内容包括任务类型、领域和标签信息。
        instruction 最终将用于提示词生成的核心指令部分。
        """
        instruction = ""
        # English
        if self.language == "en":
            # en ner task
            if self.task == "ner":
                if domain and labels:
                    instruction += NER_EN_DOMAIN_LABELS.format(
                        domain, ", ".join(labels)
                    )
                elif domain:
                    instruction += NER_EN_DOMAIN.format(domain)
                elif labels:
                    instruction += NER_EN_LABELS.format(", ".join(labels))
                else:
                    instruction += NER_EN
                instruction += NER_EN_OUTPUT
            # en re task
            elif self.task == "re":
                if domain and labels:
                    instruction += RE_EN_DOMAIN_LABELS.format(
                        domain, ", ".join(labels)
                    )
                elif domain:
                    instruction += RE_EN_DOMAIN.format(domain)
                elif labels:
                    instruction += RE_EN_LABELS.format(", ".join(labels))
                else:
                    instruction += RE_EN
                instruction += RE_EN_OUTPUT
            # en ee task
            elif self.task == "ee":
                if domain:
                    instruction += EE_EN_DOMAIN.format(domain)
                else:
                    instruction += EE_EN
                instruction += EE_EN_OUTPUT
            # en rte task
            elif self.task == "rte":
                if domain:
                    instruction += RTE_EN_DOMAIN.format(domain)
                else:
                    instruction += RTE_EN
                instruction += RTE_EN_OUTPUT
            elif self.task == "da":
                instruction += DA_EN_LABELS.format(", ".join(labels))
        # Chinese
        elif self.language == "ch":
            # ch ner task
            if self.task == "ner":
                if domain and labels:
                    instruction += NER_CH_DOMAIN_LABELS.format(
                        domain, "，".join(labels)
                    )
                elif domain:
                    instruction += NER_CH_DOMAIN.format(domain)
                elif labels:
                    instruction += NER_CH_LABELS.format("，".join(labels))
                else:
                    instruction += NER_CH
                instruction += NER_CH_OUTPUT
            # ch re task
            elif self.task == "re":
                if domain and labels:
                    instruction += RE_CH_DOMAIN_LABELS.format(
                        domain, "，".join(labels)
                    )
                elif domain:
                    instruction += RE_CH_DOMAIN.format(domain)
                elif labels:
                    instruction += RE_CH_LABELS.format("，".join(labels))
                else:
                    instruction += RE_CH
                instruction += RE_CH_OUTPUT
            # ch ee task
            elif self.task == "ee":
                if domain:
                    instruction += EE_CH_DOMAIN.format(domain)
                else:
                    instruction += EE_CH
                instruction += EE_CH_OUTPUT
            # ch rte task
            elif self.task == "rte":
                if domain:
                    instruction += RTE_CH_DOMAIN.format(domain)
                else:
                    instruction += RTE_CH
                instruction += RTE_CH_OUTPUT
            elif self.task == "da":
                instruction += DA_CH_LABELS.format("，".join(labels))
        # return
        return instruction

    def _get_incontext_examples(self, examples):
        """
        _get_incontext_examples 方法处理 in-context 示例格式，生成包含样例的提示词，用于上下文学习。
        不同任务的示例格式化不同，例如关系抽取任务包含head_entity、tail_entity等信息。
        """
        self.examples = []
        # English
        if self.language == 'en':
            if self.task == "da":
                key = f"Here are some samples for relation '{examples[0]['relation']}'"
            else:
                key = "Examples"
            self.examples.append({key: ""})
            for example in examples:
                if self.task in ["ee", "rte"]:
                    self.examples.append(
                        {"Input": example["input"], "Output": str(example["output"])}
                    )
                elif self.task == "ner":
                    self.examples.append({"Output": str(example["output"])})
                elif self.task == "re":
                    self.examples.append(
                        {
                            "Context": example["context"],
                            "Output": f"The relation between ({example['head_type']}) '{example['head_entity']}' and ({example['tail_type']}) '{example['tail_entity']}' in the context is {example['relation']}.",
                        }
                    )
                elif self.task == "da":
                    self.examples.append(
                        {
                            "Relation": example["relation"],
                            "Context": example["context"],
                            "Head Entity": example["head_entity"],
                            "Head Type": example["head_type"],
                            "Tail Entity": example["tail_entity"],
                            "Tail Type": example["tail_type"],
                        }
                    )
        # Chinese
        elif self.language == 'ch':
            if self.task == "da":
                key = f"这里有一些关于关系‘{examples[0]['relation']}’的样本"
            else:
                key = "示例"
            self.examples.append({key: ""})
            for example in examples:
                if self.task in ["ee", "rte"]:
                    self.examples.append(
                        {"输入": example["input"], "输出": str(example["output"])}
                    )
                elif self.task == "ner":
                    self.examples.append({"输出": str(example["output"])})
                elif self.task == "re":
                    self.examples.append(
                        {
                            "上下文": example["context"],
                            "输出": f"上下文头实体（{example['head_type']}）‘{example['head_entity']}’和尾实体（{example['tail_type']}）‘{example['tail_entity']}’间的关系类型是{example['relation']}",
                        }
                    )
                elif self.task == "da":
                    self.examples.append(
                        {
                            "关系": example["relation"],
                            "上下文": example["context"],
                            "头实体": example["head_entity"],
                            "头实体类型": example["head_type"],
                            "尾实体": example["tail_entity"],
                            "尾实体类型": example["tail_type"],
                        }
                    )
        # return
        return  self.examples

    def build_prompt_by_shots(
            self,
            prompt: str,
            in_context_examples: List[Union[str, Dict]] = None,
            n_shots: int = 2
    ):
        n_shots = min(len(in_context_examples), n_shots)

        context = ""
        for idx, example in enumerate(in_context_examples[:n_shots]):
            if isinstance(example, str):
                context += f"{idx+1}. {example}\n"
            elif isinstance(example, dict):
                context += f"{idx+1}."
                for key, value in example.items():
                    context += f" {key}: {value}"
                context += "\n"
            else:
                raise TypeError(
                    "in_context_examples must be a list of strings or dicts"
                )

        final_prompt = context + prompt
        print(final_prompt)
        return final_prompt



# test
if __name__ == "__main__":
    pass
