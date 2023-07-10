IGNORE_INDEX = -100

def preprocess_chatglm(example, prompter, tokenizer, options):
    model_inputs = {}
    prompt = prompter.generate_prompt(example["instruction"], example["input"])
    source_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(text=example["output"], add_special_tokens=False)

    if len(source_ids) > options.max_source_length - 2: # gmask and sop tokens
        source_ids = source_ids[:options.max_source_length - 2]
    if len(target_ids) > options.max_target_length - 1: # eos token
        target_ids = target_ids[:options.max_target_length - 1]

    context_length = len(source_ids) + 2 # gmask and sop tokens
    input_ids = tokenizer.build_inputs_with_special_tokens(source_ids, target_ids)
    labels = [IGNORE_INDEX] * context_length + input_ids[context_length:]

    model_inputs["input_ids"] = input_ids
    model_inputs["labels"] = labels
    return model_inputs


def preprocess_chatglm_eval(example, prompter, tokenizer, options):
    model_inputs = {}
    prompt, answer = prompter.generate_prompt(example)
    source_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(text=answer, add_special_tokens=False)

    if len(source_ids) > options.max_source_length - 2: # gmask and sop tokens
        source_ids = source_ids[:options.max_source_length - 2]
    if len(target_ids) > options.max_target_length - 2: # gmask and sop tokens
        target_ids = target_ids[:options.max_target_length - 2]

    input_ids = tokenizer.build_inputs_with_special_tokens(source_ids)
    labels = tokenizer.build_inputs_with_special_tokens(target_ids)

    model_inputs["input_ids"] = input_ids
    model_inputs["labels"] = labels
    return model_inputs


def preprocess_chatglm_pairwise(example, prompter, tokenizer, options):
    model_inputs = {}
    prompt, answer = prompter.generate_prompt(example)
    source_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
    accept_ids = tokenizer.encode(text=answer[0], add_special_tokens=False)
    reject_ids = tokenizer.encode(text=answer[1], add_special_tokens=False)

    if len(source_ids) > options.max_source_length - 2: # gmask and sop tokens
        source_ids = source_ids[:options.max_source_length - 2]
    if len(accept_ids) > options.max_target_length - 1: # eos token
        accept_ids = accept_ids[:options.max_target_length - 1]
    if len(reject_ids) > options.max_target_length - 1: # eos token
        reject_ids = reject_ids[:options.max_target_length - 1]

    accept_ids = tokenizer.build_inputs_with_special_tokens(source_ids[:], accept_ids) # avoid copying error
    reject_ids = tokenizer.build_inputs_with_special_tokens(source_ids[:], reject_ids)

    model_inputs["input_ids"] = accept_ids
    model_inputs["labels"] = reject_ids
    return model_inputs


def coll_fn_chatglm(stage = "sft"):
    if stage == "sft":
        return preprocess_chatglm    
    elif stage == "rm":
        return preprocess_chatglm_pairwise
    elif stage == "ppo":
        return preprocess_chatglm_eval

