"""
prompt_crafting.py

Mainly create custom prompts for those tasks to better chat with LLMs.
"""

# from openai import OpenAI
from typing import List, Dict, Union


NER_EN_DOMAIN_LABELS = "You are a highly intelligent and accurate {0} domain Named-entity recognition(NER) system. You take Passage as input and your task is to recognize and extract specific types of {0} domain named entities in that given passage and classify into a set of following predefined entity types:\n{1}\n"
NER_EN_DOMAIN = "You are a highly intelligent and accurate {0} domain Named-entity recognition(NER) system. You take Passage as input and your task is to recognize and extract specific types of {0} domain named entities in that given passage and classify into a set of entity types.\n"
NER_EN_LABELS = "You are a highly intelligent and accurate Named-entity recognition(NER) system. You take Passage as input and your task is to recognize and extract specific types of named entities in that given passage and classify into a set of following predefined entity types:\n{0}\n"
NER_EN = "You are a highly intelligent and accurate relation extraction(RE) system. Given a context, a pair of head and tail entities in the context, your task is to extract the specific type of relationship between the head and tail entities.\n"
NER_EN_OUTPUT = "Your output format is only [{'E': type of entity from predefined entity types, 'W': entity in the input text},...] form, no other form.\n\n"

RE_EN_DOMAIN_LABELS = "You are a highly intelligent and accurate {0} domain relation extraction(RE) system. Given a context, a pair of head and tail entities in the context, your task is to extract the specific type of {0} domain relationship between the head and tail entities from candidate relations:\n{1}\n"
RE_EN_DOMAIN = "You are a highly intelligent and accurate {0} domain relation extraction(RE) system. Given a context, a pair of head and tail entities in the context, your task is to extract the specific type of {0} domain relationship between the head and tail entities.\n"
RE_EN_LABELS = "You are a highly intelligent and accurate relation extraction(RE) system. Given a context, a pair of head and tail entities in the context, your task is to extract the specific type of relationship between the head and tail entities from candidate relations:\n{0}\n"
RE_EN = "You are a highly intelligent and accurate relation extraction(RE) system. Given a context, a pair of head and tail entities in the context, your task is to extract the specific type of relationship between the head and tail entities.\n"
RE_EN_OUTPUT = "Your output is only the relation type, no other words.\n\n"

EE_EN_DOMAIN = "You are a highly intelligent and accurate {0} domain event extraction model. You take Passage as input and convert it into {0} domain events arguments.You can identify all events of target types mentioned in the sentence, and extract corresponding event arguments playing target roles.\n"
EE_EN = "You are a highly intelligent and accurate event extraction model. You take Passage as input and convert it into events arguments. You can identify all events of target types mentioned in the sentence, and extract corresponding event arguments playing target roles.\n"
EE_EN_OUTPUT = "Your output format is only [{event_type, arguments: [{role , argument}, ...]}, ...], no other form.\n\n"

RTE_EN_DOMAIN = "You are a highly intelligent and accurate {0} domain Resource Description Framework (RDF) data model. You take Passage as input and convert it into {0} domain RDF triples. A triple is a set of three entities that codifies a statement about semantic data in the form of subject-predicate-object expressions.\n"
RTE_EN = "You are a highly intelligent and accurate Resource Description Framework (RDF) data model. You take Passage as input and convert it into RDF triples. A triple is a set of three entities that codifies a statement about semantic data in the form of subject-predicate-object expressions.\n"
RTE_EN_OUTPUT = "Your output format is only [[ subject, predicate, object ], ...], no other form.\n\n"

DA_EN_LABELS = "One sample in relation extraction datasets consists of a relation, a context, a pair of head and tail entities in the context and their entity types. The head entity has the relation with the tail entity and entities are pre-categorized as the following types: \n{0}\n"


NER_CH_DOMAIN_LABELS = "您是一个高度智能和精确的{0}域命名实体识别（NER）系统。您将文本作为输入，您的任务是识别和提取给定段落中的特定类型的{0}域名实体，并将其分类为一组预定义的实体类型：\n{1}\n"
NER_CH_DOMAIN = "您是一个高度智能和精确的{0}域命名实体识别（NER）系统。您将文本作为输入，您的任务是识别和提取给定文章中特定类型的{0}域命名实体，并将其分类为一组实体类型。\n"
NER_CH_LABELS = "您是一个高度智能和精确的命名实体识别（NER）系统。您将文本作为输入，您的任务是识别和提取给定段落中的特定类型的命名实体，并将其分类为一组预定义的实体类型：\n{0}\n"
NER_CH = "您是一个高度智能和精确的命名实体识别（NER）系统。您将文本作为输入，您的任务是识别和提取给定文章中特定类型的命名实体，并将其分类为一组实体类型。\n"
NER_CH_OUTPUT = "您输出的格式需要为：[{'E': 预先定义的实体类型, 'W': 输入文本中的实体},...]，没有其他格式要求。\n\n"

RE_CH_DOMAIN_LABELS = "您是一个高度智能和精确的{0}域关系抽取（RE）系统。给定上下文以及上下文中包含的一对头实体和尾实体，您的任务是提取给定头实体和尾实体间特定类型的{0}域关系，候选的关系类型如下：\n{1}\n"
RE_CH_DOMAIN = (
    "您是一个高度智能和精确的{0}域关系抽取（RE）系统。给定上下文以及上下文中包含的一对头实体和尾实体，您的任务是提取给定头实体和尾实体间特定类型的{0}域关系。\n"
)
RE_CH_LABELS = "您是一个高度智能和精确的关系抽取（RE）系统。给定上下文以及上下文中包含的一对头实体和尾实体，您的任务是提取给定头实体和尾实体间特定类型的关系，候选的关系类型如下：\n{0}\n"
RE_CH = "您是一个高度智能和精确的关系抽取（RE）系统。给定上下文以及上下文中包含的一对头实体和尾实体，您的任务是提取给定头实体和尾实体间特定类型的关系。\n"
RE_CH_OUTPUT = "您只需要输出关系的类型即可，不需要其他的文字输出。\n\n"

EE_CH_DOMAIN = "您是一个高度智能和精确的{0}域事件提取模型。您将文本作为输入并将其转换为{0}域事件参数。您可以识别句子中提到的所有目标类型的事件，并提取扮演目标角色的相应事件参数。\n"
EE_CH = "您是一个高度智能和精确的事件提取模型。您将文本作为输入并将其转换为事件参数。您可以识别句子中提到的所有目标类型的事件，并提取扮演目标角色的相应事件参数。\n"
EE_CH_OUTPUT = (
    "您的输出全部为中文，且格式为：[{事件类型, 参数: [{角色, 参数内容}, ...]}, ...]，没有其他格式要求。\n\n"
)

RTE_CH_DOMAIN = "您是一个高度智能和精确的{0}域资源描述框架（RDF）数据模型。您将文本作为输入，并将其转换为{0}域RDF三元组。三元组是由三个实体组成的集合，以主语-谓语-宾语表达式的形式对语义数据进行编码。\n"
RTE_CH = "您是一个高度智能和精确的资源描述框架（RDF）数据模型。您将文本作为输入，并将其转换为RDF三元组。三元组是由三个实体组成的集合，以主语-谓语-宾语表达式的形式对语义数据进行编码。\n"
RTE_CH_OUTPUT = "您输出的格式需要为：[[ 主语, 谓语, 宾语 ], ...]，没有其他格式要求。\n\n"

DA_CH_LABELS = (
    "关系提取数据集中的一个样本由关系、文本、文本中的一对头实体和尾实体及它们的实体类型组成。头实体与尾实体间存在关系，依次被预先分类为以下类型：\n{0}\n"
)


class PromptCraft:
    """
    A class to create custom prompts for various NLP tasks.

    This class generates prompts based on user-defined parameters and task requirements,
    supporting tasks such as named entity recognition (NER), relation extraction (RE),
    event extraction (EE), relation triplet extraction (RTE), and data augmentation (DA).
    """
    def __init__(
            self,
            task: str = 'ner',
            language: str = 'en',
            in_context: bool = True,
            instruction: str = None,
            example: Dict = None
    ):
        """
        Initializes the PromptCraft with task details and settings.

        :param task: The type of NLP task (e.g., 'ner', 're', 'ee', 'rte', 'da').
        :param language: The language for the task ('en' for English, 'ch' for Chinese).
        :param in_context: Indicates whether to use in-context examples.
        :param instruction: Custom instruction for the prompt (if any).
        :param example: Examples to use for in-context learning.
        """
        self.task = task
        self.language = language
        self.in_context = in_context
        self.instruction = instruction
        self.examples = example

    def build_prompt(
            self,
            prompt: str,
            domain: str = None,
            labels: List = None,
            head_entity: str = None,
            head_type: str = None,
            tail_entity: str = None,
            tail_type: str = None,
    ) -> str:
        """
        Constructs the final prompt based on the provided inputs and task parameters.

        :param prompt: The initial prompt input by the user.
        :param domain: The domain of the input text (if applicable).
        :param labels: List of labels relevant to the task (if applicable).
        :param head_entity: The head entity for relation extraction tasks.
        :param head_type: The type of the head entity for relation extraction tasks.
        :param tail_entity: The tail entity for relation extraction tasks.
        :param tail_type: The type of the tail entity for relation extraction tasks.
        :return: The constructed prompt ready for model input.
        """
        # <01: check>
        if self.task == "da" and labels is None:
            raise ValueError("Please provide some pre-categorized entity types if the task is Data Augmentation(da).")

        # <02: generate>
        # instruction
        if self.instruction is None:
            self.instruction = self._get_default_instruction(domain, labels)
        else:
            self.instruction += '\n'
        # pre-prompt for 're' & 'da'
        if self.language == 'en':
            if self.task == "re":
                prompt = f"Context: {prompt}\nThe relation between ({head_type}) '{head_entity}' and ({tail_type}) '{tail_entity}' in the context is: "
            elif self.task == "da":
                prompt = f"Generate more samples for the relation '{prompt}':\n"
            else:
                prompt = f"Input: {prompt}\nOutput: "
        elif self.language == 'ch':
            if self.task == "re":
                prompt = f"上下文：{prompt}\n上下文中头实体（{head_type}）‘{head_entity}’和尾实体（{tail_type}）‘{tail_entity}’之间的关系类型是："
            elif self.task == "da":
                prompt = f"请为关系‘{prompt}’生成更多的样例数据：\n"
            else:
                prompt = f"输入：{prompt}\n输出："
        # in_context examples
        if self.in_context:
            final_examples = self._get_incontext_examples(self.examples)
            self.examples = final_examples # update self.examples
            final_prompt = self._build_prompt_by_shots(
                prompt=prompt,
                in_context_examples=final_examples,
                n_shots=len(final_examples)
            )
        else:
            final_prompt = prompt

        # <03: final get>
        final_prompt = self.instruction + final_prompt
        return final_prompt

    def _get_default_instruction(self, domain, labels):
        """
        Generates default instructions based on the task and provided parameters.

        :param domain: The domain for the task (if applicable).
        :param labels: List of labels relevant to the task (if applicable).
        :return: A string containing the default instruction.
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
        # print("Auto-gen Instruction: " + instruction) # test
        return instruction

    def _get_incontext_examples(self, examples):
        """
        Retrieves and formats in-context examples based on the task and language.

        :param examples: A list of example data to format for in-context learning.
        :return: A list of formatted in-context examples.
        """
        final_examples = []
        # English
        if self.language == 'en':
            if self.task == "da":
                key = f"Here are some samples for relation '{examples[0]['relation']}'"
            else:
                key = "Examples"
            final_examples.append({key: ""})
            for example in examples:
                if self.task in ["ee", "rte"]:
                    final_examples.append(
                        {"Input": example["input"], "\nOutput": str(example["output"])}
                    )
                elif self.task == "ner":
                    final_examples.append(
                        {"Input": example["input"], "\nOutput": str(example["output"])}
                    )
                elif self.task == "re":
                    final_examples.append(
                        {
                            "Context": example["context"],
                            "Output": f"The relation between ({example['head_type']}) '{example['head_entity']}' and ({example['tail_type']}) '{example['tail_entity']}' in the context is: {example['relation']}",
                        }
                    )
                elif self.task == "da":
                    final_examples.append(
                        {
                            "Relation": example["relation"],
                            "\nContext": example["context"],
                            "\nHead Entity": example["head_entity"],
                            "\nHead Type": example["head_type"],
                            "\nTail Entity": example["tail_entity"],
                            "\nTail Type": example["tail_type"],
                        }
                    )
        # Chinese
        elif self.language == 'ch':
            if self.task == "da":
                key = f"这里有一些关于关系‘{examples[0]['relation']}’的样本"
            else:
                key = "示例"
            final_examples.append({key: ""})
            for example in examples:
                if self.task in ["ee", "rte"]:
                    final_examples.append(
                        {"输入": example["input"], "\n输出": str(example["output"])}
                    )
                elif self.task == "ner":
                    final_examples.append(
                        {"输入": example["input"], "\n输出": str(example["output"])}
                    )
                elif self.task == "re":
                    final_examples.append(
                        {
                            "上下文": example["context"],
                            "输出": f"上下文头实体（{example['head_type']}）‘{example['head_entity']}’和尾实体（{example['tail_type']}）‘{example['tail_entity']}’间的关系类型是： {example['relation']}",
                        }
                    )
                elif self.task == "da":
                    final_examples.append(
                        {
                            "关系": example["relation"],
                            "\n上下文": example["context"],
                            "\n头实体": example["head_entity"],
                            "\n头实体类型": example["head_type"],
                            "\n尾实体": example["tail_entity"],
                            "\n尾实体类型": example["tail_type"],
                        }
                    )
        # return
        # print("Your in-context examples: ", final_examples) # test
        return final_examples

    def _build_prompt_by_shots(
            self,
            prompt: str,
            in_context_examples: List[Union[str, Dict]] = None,
            n_shots: int = 2,
    ) -> str:
        """
        Builds a prompt that incorporates in-context examples into the final prompt.

        :param prompt: The initial prompt to which examples will be added.
        :param in_context_examples: A list of examples to use for in-context learning.
        :param n_shots: The number of examples to include in the prompt.
        :return: The final prompt that includes in-context examples.
        """
        n_shots = min(len(in_context_examples), n_shots)

        context = ""
        for example in in_context_examples[:n_shots]:
            if isinstance(example, str):
                context += f"{example}\n"
            elif isinstance(example, dict):
                for key, value in example.items():
                    context += f"{key}: {value}"
                context += "\n"
            else:
                raise TypeError(
                    "in_context_examples must be a list of strings or dicts"
                )

        final_prompt = context + prompt
        # print("Your prompt build by shots: " + final_prompt) # test
        return final_prompt

    def get_llm_result(self, ):
        pass
        # client = OpenAI()
        # self.engine = engine
        # if engine in API_NAME_DICT["openai"]["gpt-3"]:
        #     response = client.completions.create(
        #         model=engine,
        #         prompt=self.prompt,
        #         temperature=temperature,
        #         max_tokens=max_tokens,
        #         top_p=top_p,
        #         n=n,
        #         frequency_penalty=frequency_penalty,
        #         presence_penalty=presence_penalty,
        #     )
        #     output = response.choices[0].text.strip()
        #
        # elif (
        #     engine in API_NAME_DICT["openai"]["gpt-3.5"]
        #     or engine in API_NAME_DICT["openai"]["gpt-4"]
        # ):
        #     if isinstance(system_message, str):
        #         messages = [
        #             {"role": "system", "content": system_message},
        #             {"role": "user", "content": self.prompt},
        #         ]
        #     elif isinstance(system_message, list):
        #         messages = system_message
        #     else:
        #         raise ValueError(
        #             "system_message should be either a string or a list of strings."
        #         )
        #
        #     response = client.chat.completions.create(
        #         model=engine,
        #         messages=messages,
        #         temperature=temperature,
        #         max_tokens=max_tokens,
        #         top_p=top_p,
        #         n=n,
        #         frequency_penalty=frequency_penalty,
        #         presence_penalty=presence_penalty,
        #     )
        #     output = response.choices[0].message.content.strip()
        #
        # else:
        #     print("[ERROR] Engine {engine} not found!".format(engine=engine))
        #     print("Available engines are as follows:")
        #     print(API_NAME_DICT["openai"])
        #     response = None
        #     output = None
        #
        # self.response = response
        # self.output = output
        #
        # return self.output


if __name__ == "__main__":
    pass