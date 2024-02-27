from convert.utils.constant import NER, RE, SPO, EE, EEA, EET, KG


instruction_mapper = {
    NER + 'zh': "你是专门进行实体抽取的专家。请从input中抽取出符合schema定义的实体，不存在的实体类型返回空列表。请按照JSON字符串的格式回答。",
    RE + 'zh': "你是专门进行关系抽取的专家。请从input中抽取出符合schema定义的关系三元组，不存在的关系返回空列表。请按照JSON字符串的格式回答。",
    EE + 'zh': "你是专门进行事件提取的专家。请从input中抽取出符合schema定义的事件，不存在的事件返回空列表，不存在的论元返回NAN，如果论元存在多值请返回列表。请按照JSON字符串的格式回答。",
    EET + 'zh': "你是专门进行事件提取的专家。请从input中抽取出符合schema定义的事件类型及事件触发词，不存在的事件返回空列表。请按照JSON字符串的格式回答。",
    EEA + 'zh': "你是专门进行事件论元提取的专家。请从input中抽取出符合schema定义的事件论元及论元角色，不存在的论元返回NAN或空字典，如果论元存在多值请返回列表。请按照JSON字符串的格式回答。",
        
    NER + 'en': "You are an expert in named entity recognition. Please extract entities that match the schema definition from the input. Return an empty list if the entity type does not exist. Please respond in the format of a JSON string.",
    RE + 'en': "You are an expert in relationship extraction. Please extract relationship triples that match the schema definition from the input. Return an empty list for relationships that do not exist. Please respond in the format of a JSON string.",
    EE + 'en': "You are an expert in event extraction. Please extract events from the input that conform to the schema definition. Return an empty list for events that do not exist, and return NAN for arguments that do not exist. If an argument has multiple values, please return a list. Respond in the format of a JSON string.",
    EET + 'en': "You are an expert in event extraction. Please extract event types and event trigger words from the input that conform to the schema definition. Return an empty list for non-existent events. Please respond in the format of a JSON string.",
    EEA + 'en': "You are an expert in event argument extraction. Please extract event arguments and their roles from the input that conform to the schema definition, which already includes event trigger words. If an argument does not exist, return NAN or an empty dictionary. Please respond in the format of a JSON string.", 
}

def get_schema_type(task, schema):
    if task == NER or task == RE or task == EET:
        return schema
    elif task == SPO:
        return f"{schema['subject_type']}_{schema['predicate']}_{schema['object_type']}"
    elif task == EE or task == EEA:
        return schema['event_type']
    elif task == KG:
        return schema['entity_type']


def get_label(task, schema):
    if task == NER or task == RE or task == EET:
        return schema
    elif task == SPO:
        return schema['predicate']
    elif task == EE or task == EEA:
        return schema['event_type']
    elif task == KG:
        return schema['entity_type']
