import os, json, pickle, logging, pprint, random
import numpy as np
from deepke.event_extraction.standard.degree.data import EEDataset
from deepke.event_extraction.standard.degree.utils import generate_vocabs
from deepke.event_extraction.standard.degree.template_generate_ace import eve_template_generator
from transformers import AutoTokenizer
import ipdb
import hydra
from hydra import utils
from omegaconf import OmegaConf, open_dict

# configuration
@hydra.main(config_path="./conf", config_name="config.yaml")
def main(config):       

    cwd = utils.get_original_cwd()
    OmegaConf.set_struct(config, True)
    with open_dict(config):
        config["cwd"] = cwd
    config.cache_dir = os.path.join(config.cwd, config.cache_dir)
    config.output_dir = os.path.join(config.cwd, config.output_dir)
    config.finetune_dir = os.path.join(config.cwd, config.finetune_dir)
    config.train_file = os.path.join(config.cwd, config.train_file)
    config.dev_file = os.path.join(config.cwd, config.dev_file)
    config.test_file = os.path.join(config.cwd, config.test_file)
    config.train_finetune_file = os.path.join(config.cwd, config.train_finetune_file)
    config.dev_finetune_file = os.path.join(config.cwd, config.dev_finetune_file)
    config.test_finetune_file = os.path.join(config.cwd, config.test_finetune_file)

    # fix random seed
    random.seed(config.seed)
    np.random.seed(config.seed)

    # logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s',
                        datefmt='[%Y-%m-%d %H:%M:%S]')
    logger = logging.getLogger(__name__)
    logger.info(f"\n{pprint.pformat(vars(config), indent=4)}")

    def generate_data(data_set, vocab, config):
        inputs = []
        targets = []
        events = []

        for data in data_set.data:
            event_template = eve_template_generator(data.tokens, data.triggers, data.roles, config.input_style,
                                                    config.output_style, vocab, True, data.knowledge)
            all_data_ = event_template.get_training_data()
            pos_data_ = [dt for dt in all_data_ if dt[3]]
            neg_data_ = [dt for dt in all_data_ if not dt[3]]
            np.random.shuffle(neg_data_)

            for data_ in pos_data_:
                inputs.append(data_[0])
                targets.append(data_[1])
                events.append((data_[2], data_[4], data_[5]))

            for data_ in neg_data_[:config.n_negative]:
                inputs.append(data_[0])
                targets.append(data_[1])
                events.append((data_[2], data_[4], data_[5]))

        return inputs, targets, events

    # check valid styles
    assert np.all([style in ['event_type_sent', 'keywords', 'template'] for style in config.input_style])
    assert np.all([style in ['trigger:sentence', 'argument:sentence'] for style in config.output_style])

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=config.cache_dir)
    special_tokens = ['<Trigger>', '<sep>']
    tokenizer.add_tokens(special_tokens)

    if not os.path.exists(config.finetune_dir):
        os.makedirs(config.finetune_dir)

    # load data
    train_set = EEDataset(tokenizer, config.train_file, max_length=config.max_length)
    dev_set = EEDataset(tokenizer, config.dev_file, max_length=config.max_length)
    test_set = EEDataset(tokenizer, config.test_file, max_length=config.max_length)
    vocab = generate_vocabs([train_set, dev_set, test_set])

    # save vocabulary
    with open('{}/vocab.json'.format(config.finetune_dir), 'w') as f:
        json.dump(vocab, f, indent=4)

        # generate finetune data
    train_inputs, train_targets, train_events = generate_data(train_set, vocab, config)
    logger.info(f"Generated {len(train_inputs)} training examples from {len(train_set)} instance")

    with open('{}/train_input.json'.format(config.finetune_dir), 'w') as f:
        json.dump(train_inputs, f, indent=4)

    with open('{}/train_target.json'.format(config.finetune_dir), 'w') as f:
        json.dump(train_targets, f, indent=4)

    with open('{}/train_all.pkl'.format(config.finetune_dir), 'wb') as f:
        pickle.dump({
            'input': train_inputs,
            'target': train_targets,
            'all': train_events
        }, f)

    dev_inputs, dev_targets, dev_events = generate_data(dev_set, vocab, config)
    logger.info(f"Generated {len(dev_inputs)} dev examples from {len(dev_set)} instance")

    with open('{}/dev_input.json'.format(config.finetune_dir), 'w') as f:
        json.dump(dev_inputs, f, indent=4)

    with open('{}/dev_target.json'.format(config.finetune_dir), 'w') as f:
        json.dump(dev_targets, f, indent=4)

    with open('{}/dev_all.pkl'.format(config.finetune_dir), 'wb') as f:
        pickle.dump({
            'input': dev_inputs,
            'target': dev_targets,
            'all': dev_events
        }, f)

    test_inputs, test_targets, test_events = generate_data(test_set, vocab, config)
    logger.info(f"Generated {len(test_inputs)} test examples from {len(test_set)} instance")

    with open('{}/test_input.json'.format(config.finetune_dir), 'w') as f:
        json.dump(test_inputs, f, indent=4)

    with open('{}/test_target.json'.format(config.finetune_dir), 'w') as f:
        json.dump(test_targets, f, indent=4)

    with open('{}/test_all.pkl'.format(config.finetune_dir), 'wb') as f:
        pickle.dump({
            'input': test_inputs,
            'target': test_targets,
            'all': test_events
        }, f)

if __name__ == '__main__':
    main()