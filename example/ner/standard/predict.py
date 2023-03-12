from deepke.name_entity_re.standard import *
import hydra
from hydra import utils
import pickle
import os

@hydra.main(config_path="conf", config_name='config')
def main(cfg):
    if cfg.model_name == 'lstmcrf':
        with open(os.path.join(utils.get_original_cwd(), cfg.data_dir, cfg.model_vocab_path), 'rb') as inp:
            word2id = pickle.load(inp)
            label2id = pickle.load(inp)
            id2label = pickle.load(inp)

        model = InferNer(utils.get_original_cwd() + '/' + "checkpoints/", cfg, len(word2id), len(label2id), word2id, id2label)
    elif cfg.model_name == 'bert':
        model = InferNer(os.path.join(utils.get_original_cwd(), "checkpoints"), cfg)
    else:
        raise NotImplementedError(f"model type {cfg.model_name} not supported")
    text = cfg.text

    print("NER句子:")
    print(text)
    print('NER结果:')

    result = model.predict(text)
    print(result)
    # for k,v in result.items():
    #     if v:
    #         print(v,end=': ')
    #         if k=='PER':
    #             print('Person')
    #         elif k=='LOC':
    #             print('Location')
    #         elif k=='ORG':
    #             print('Organization')
   
    
if __name__ == "__main__":
    main()
