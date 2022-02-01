import os
import hydra
import torch
import numpy as np
import random
from torchvision import transforms
from hydra import utils
from torch.utils.data import DataLoader
from deepke.relation_extraction.multimodal.models.IFA_model import IFAREModel
from deepke.relation_extraction.multimodal.modules.dataset import MMREProcessor, MMREDataset
from deepke.relation_extraction.multimodal.modules.train import Trainer

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

import wandb
writer = wandb.init(project="DeepKE_RE_MM")

DATA_PATH = {
        'test': 'data/txt/ours_test.txt',
        'test_auximgs': 'data/txt/mre_test_dict.pth',
        'test_img2crop': 'data/img_detect/test/test_img2crop.pth'}

IMG_PATH = {
    'test': 'data/img_org/test'}

AUX_PATH = {
        'test': 'data/img_vg/test/crops'}

re_path = 'data/ours_rel2id.json'

def set_seed(seed=2021):
    """set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

@hydra.main(config_path="conf/config.yaml")
def main(cfg):
    cwd = utils.get_original_cwd()
    cfg.cwd = cwd
    print(cfg)
   
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    set_seed(cfg.seed) # set seed, default is 1
    if cfg.save_path is not None:  # make save_path dir
        if not os.path.exists(cfg.save_path):
            os.makedirs(cfg.save_path, exist_ok=True)
    print(cfg)

    processor = MMREProcessor(DATA_PATH, re_path, cfg)
    test_dataset = MMREDataset(processor, transform, IMG_PATH, AUX_PATH, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    re_dict = processor.get_relation_dict()
    num_labels = len(re_dict)
    tokenizer = processor.tokenizer

    model = IFAREModel(num_labels, tokenizer, cfg)

    trainer = Trainer(train_data=None, dev_data=None, test_data=test_dataloader, re_dict=re_dict, model=model, args=cfg, logger=logger, writer=writer)
    trainer.predict()


if __name__ == '__main__':
    main()