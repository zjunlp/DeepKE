import os
import torch
from transformers import DataCollatorForSeq2Seq, Trainer
from dataclasses import dataclass



@dataclass
class ChatGLMDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):

    def __call__(self, features):
        len_ids = [len(feature["input_ids"]) for feature in features]
        longest = max(len_ids)
        input_ids = []
        labels_list = []
        for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: - x[0]):
            ids = feature["input_ids"]
            seq_len = feature["seq_len"]
            labels = (
                [-100] * (seq_len - 1) + ids[(seq_len - 1):] + [-100] * (longest - ids_l)
            )
            ids = ids + [self.tokenizer.pad_token_id] * (longest - ids_l)
            _ids = torch.LongTensor(ids)
            labels_list.append(torch.LongTensor(labels))
            input_ids.append(_ids)

        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels_list)
        results = {"input_ids": input_ids, "labels": labels}

        return results


def coll_fn_glm(example, prompter, tokenizer, options):
    prompt = prompter.generate_prompt(
        example["instruction"],
        example["input"],
    )
    prompt_ids = tokenizer.encode(
        prompt,
        truncation=True,
        max_length=options.cutoff_len
    )
    target_ids = tokenizer.encode(
        example["output"],
        max_length=options.cutoff_len,
        truncation=True,
        add_special_tokens=False
    )
    input_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
    input_ids = input_ids[:options.cutoff_len]
    result = {"input_ids": input_ids[:options.cutoff_len], "seq_len": len(prompt_ids), }

    return result



class ChatGLMTrainer(Trainer):
    def compute_loss(self, model, inputs):
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss

    def save_model(self, output_dir=None):
        from transformers.trainer import TRAINING_ARGS_NAME

        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))