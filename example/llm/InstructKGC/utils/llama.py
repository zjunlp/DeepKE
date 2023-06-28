from dataclasses import dataclass

def llama_tokenize(prompt, tokenizer, options):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=options.cutoff_len,
        padding=False,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < options.cutoff_len
        and options.add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
    result["labels"] = result["input_ids"].copy()

    return result


def coll_fn_llama(example, prompter, tokenizer, options):
    full_prompt = prompter.generate_prompt(
        example["instruction"],
        example["input"],
        example["output"],
    )
    tokenized_full_prompt = llama_tokenize(full_prompt, tokenizer, options)
    if not options.train_on_inputs:
        user_prompt = prompter.generate_prompt(
            example["instruction"], example["input"]
        )
        tokenized_user_prompt = llama_tokenize(user_prompt, tokenizer, options)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        if options.add_eos_token:
            user_prompt_len -= 1

        tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:] 
    return tokenized_full_prompt
    