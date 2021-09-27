import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import argparse
import logging
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from torch.utils.data import DataLoader
from deepke.name_entity_re.few_shot.models.model import PromptBartModel, PromptGeneratorModel
from deepke.name_entity_re.few_shot.modules.datasets import ConllNERProcessor, ConllNERDataset
from deepke.name_entity_re.few_shot.modules.train import Trainer
from deepke.name_entity_re.few_shot.modules.metrics import Seq2SeqSpanMetric
from deepke.name_entity_re.few_shot.utils.utils import get_loss, set_seed
from deepke.name_entity_re.few_shot.modules.mapping_type import mit_movie_mapping, mit_restaurant_mapping, atis_mapping

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from tensorboardX import SummaryWriter
writer = SummaryWriter(log_dir='logs')

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


DATASET_CLASS = {
    'conll2003': ConllNERDataset,
    'mit-movie': ConllNERDataset,
    'mit-restaurant': ConllNERDataset,
    'atis': ConllNERDataset
}

DATA_PROCESS = {
    'conll2003': ConllNERProcessor,
    'mit-movie': ConllNERProcessor,
    'mit-restaurant': ConllNERProcessor,
    'atis': ConllNERProcessor
}

DATA_PATH = {
    'conll2003': {'train': 'data/conll2003/train.txt',
                  'dev': 'data/conll2003/dev.txt',
                  'test': 'data/conll2003/test.txt'},
    'mit-movie': {'train': 'data/mit-movie/20-shot-train.txt',
                  'dev': 'data/mit-movie/test.txt'},
    'mit-restaurant': {'train': 'data/mit-restaurant/10-shot-train.txt',
                  'dev': 'data/mit-restaurant/test.txt'},
    'atis': {'train': 'data/atis/20-shot-train.txt',
                  'dev': 'data/atis/test.txt'}
}

MAPPING = {
    'conll2003': {'loc': '<<location>>',
                'per': '<<person>>',
                'org': '<<organization>>',
                'misc': '<<others>>'},
    'mit-movie': mit_movie_mapping,
    'mit-restaurant': mit_restaurant_mapping,
    'atis': atis_mapping
}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', default='conll2003', type=str, help="The name of dataset, choose from [conll2003, mit-movie, mit-restaurant, atis].")
    parser.add_argument('--bart_name', default='bart-large', type=str, help="Pretrained language model name, bart-base or bart-large")
    parser.add_argument('--num_epochs', default=30, type=int, help="Training epochs")
    parser.add_argument('--device', default='cuda', type=str, help="cuda or cpu")
    parser.add_argument('--batch_size', default=16, type=int, help="batch size")
    parser.add_argument('--learning_rate', default=2e-5, type=float, help="learning rate")
    parser.add_argument('--warmup_ratio', default=0.01, type=float)
    parser.add_argument('--eval_begin_epoch', default=16, type=int)
    parser.add_argument('--num_beams', default=1, type=int, help="beam search size")
    parser.add_argument('--seed', default=1, type=int, help="random seed, default is 1")
    parser.add_argument('--length_penalty', default=1, type=int)
    parser.add_argument('--tgt_max_len', default=10, type=int)
    parser.add_argument('--src_seq_ratio', default=0.6, type=float)
    parser.add_argument('--prompt_len', default=10, type=int)
    parser.add_argument('--prompt_dim', default=800, type=int)
    parser.add_argument('--load_path', default=None, type=str, help="Load model from load_path")
    parser.add_argument('--save_path', default=None, type=str, help="save model at save_path")
    parser.add_argument('--write_path', default=None, type=str, help="do_test=True, predictions will be write in write_path")
    parser.add_argument('--notes', default="", type=str, help="input some remarks for making save path dir.")
    parser.add_argument('--use_prompt', action='store_true')
    parser.add_argument('--learn_weights', action='store_true', help="If true, don't use extra tokens, the embeddings corresponding to the entity labels are combined with learnable weights.")
    parser.add_argument('--freeze_plm', action='store_true')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')

    args = parser.parse_args()

    data_path = DATA_PATH[args.dataset_name]
    dataset_class, data_process = DATASET_CLASS[args.dataset_name], DATA_PROCESS[args.dataset_name]
    mapping = MAPPING[args.dataset_name]

    set_seed(args.seed) # set seed, default is 1
    if args.save_path is not None:  # make save_path dir
        args.save_path = os.path.join(args.save_path, args.dataset_name+"_"+str(args.batch_size)+"_"+str(args.learning_rate)+args.notes)
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path, exist_ok=True)
    print(args)
    if args.do_train:
        process = data_process(data_path=data_path, mapping=mapping, bart_name=args.bart_name, learn_weights=args.learn_weights)
        train_dataset = dataset_class(data_processor=process, mode='train')
        train_dataloader = DataLoader(train_dataset, collate_fn=train_dataset.collate_fn, batch_size=args.batch_size, num_workers=4)
        
        dev_dataset = dataset_class(data_processor=process, mode='dev')
        dev_dataloader = DataLoader(dev_dataset, collate_fn=dev_dataset.collate_fn, batch_size=args.batch_size, num_workers=4)

        label_ids = list(process.mapping2id.values())

        prompt_model = PromptBartModel(tokenizer=process.tokenizer, label_ids=label_ids, args=args)
        model = PromptGeneratorModel(prompt_model=prompt_model, bos_token_id=0,
                                    eos_token_id=1,
                                    max_length=args.tgt_max_len, max_len_a=args.src_seq_ratio,num_beams=args.num_beams, do_sample=False,
                                    repetition_penalty=1, length_penalty=args.length_penalty, pad_token_id=1,
                                    restricter=None)
        metrics = Seq2SeqSpanMetric(eos_token_id=1, num_labels=len(label_ids), target_type='word')
        loss = get_loss

        trainer = Trainer(train_data=train_dataloader, dev_data=dev_dataloader, test_data=None, model=model, args=args, logger=logger, loss=loss, metrics=metrics, writer=writer)
        trainer.train()

        writer.close()
    elif args.do_test:
        process = data_process(data_path=data_path, mapping=mapping, bart_name=args.bart_name, learn_weights=args.learn_weights)
        test_dataset = dataset_class(data_processor=process, mode='test')
        test_dataloader = DataLoader(test_dataset, collate_fn=test_dataset.collate_fn, batch_size=args.batch_size, num_workers=4)

        label_ids = list(process.mapping2id.values())
        prompt_model = PromptBartModel(tokenizer=process.tokenizer, label_ids=label_ids, args=args)
        model = PromptGeneratorModel(prompt_model=prompt_model, bos_token_id=0,
                                    eos_token_id=1,
                                    max_length=args.tgt_max_len, max_len_a=args.src_seq_ratio,num_beams=args.num_beams, do_sample=False,
                                    repetition_penalty=1, length_penalty=args.length_penalty, pad_token_id=1,
                                    restricter=None)
        trainer = Trainer(train_data=None, dev_data=None, test_data=test_dataloader, model=model, process=process, args=args, logger=logger, loss=None, metrics=None, writer=writer)
        trainer.predict()


if __name__ == "__main__":
    main()