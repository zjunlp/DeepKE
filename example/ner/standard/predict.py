from deepke.name_entity_re.standard import *
import hydra
from hydra import utils

@hydra.main(config_path="conf", config_name='config')
def main(cfg):
    model = InferNer(utils.get_original_cwd()+'/'+"checkpoints/")
    text = cfg.text

    print("NER句子:")
    print(text)
    print('NER结果:')

    result = model.predict(text)
    for k,v in result.items():
        if v:
            print(v,end=': ')
            if k=='PER':
                print('Person')
            elif k=='LOC':
                print('Location')
            elif k=='ORG':
                print('Organization')
   
    
if __name__ == "__main__":
    main()
