
def llama_tokenize(prompt, tokenizer, cutoff_len, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
    result["labels"] = result["input_ids"].copy()
    return result


def preprocess_llama(example, prompter, tokenizer, options):
    full_prompt = prompter.generate_prompt(
        example["instruction"],
        example["input"],
        example["output"],
    )
    tokenized_full_prompt = llama_tokenize(full_prompt, tokenizer, options.cutoff_len)
    if not options.train_on_inputs:
        user_prompt = prompter.generate_prompt(
            example["instruction"], example["input"]
        )
        tokenized_user_prompt = llama_tokenize(user_prompt, tokenizer, options.cutoff_len, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
        
    return tokenized_full_prompt
    

def coll_fn_llama(stage = "sft"):
    return preprocess_llama
