import os
from easyinstruct import BasePrompt
import json

os.environ["https_proxy"] = "http://127.0.0.1:7890/"


def codekgc(config_path: str = 'config.json'):
    with open(config_path, 'r') as f:
        config = json.load(f)
    [schema_path, ICL_path, example_path, openai_key, engine, temperature, max_tokens, n] = [config["schema_path"], config["ICL_path"], config["example_path"], config["openai_key"], config["engine"], config["temperature"], config["max_tokens"], config["n"]]
    
    os.environ['OPENAI_API_KEY'] = openai_key

    ''' 
        code prompt
    '''
    with open(schema_path, 'r') as f:
        schema_prompt = f.read()
    with open(ICL_path, 'r') as f:
        ICL_prompt = f.read()
    
    ''' 
        test example
    '''
    with open(example_path, 'r') as f:
        item = json.load(f)

    ''' 
        call the api
    ''' 
    text = item['text']
    labels = item['triple_list']

    text = '\n""" ' + text + ' """'
    prompt = schema_prompt + '\n' + ICL_prompt +  text
    print(prompt)
    
    code_prompt = BasePrompt()
    code_prompt.build_prompt(prompt)
    completion = code_prompt.get_openai_result(
        engine = engine, 
        temperature = temperature,
        max_tokens = max_tokens,
        n = n
        )
    result = completion.choices[0].text.strip()
    print(result)

if __name__ == "__main__":
    codekgc()