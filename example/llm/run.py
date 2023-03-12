

# # Define the API key for the OpenAI model


from model import OpenAI
from prompt import Prompter
import json
import os
import hydra
from hydra import utils
import logging
logger = logging.getLogger(__name__)

# sentence     =  """The patient is a 93-year-old female with a medical  				 
#                 history of chronic right hip pain, osteoporosis,					
#                 hypertension, depression, and chronic atrial						
#                 fibrillation admitted for evaluation and management				
#                 of severe nausea and vomiting and urinary tract				
#                 infection"""


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):

  cfg.cwd = utils.get_original_cwd()

  text = cfg.text_input

  if not cfg.api_key:
    raise ValueError("Need an API Key.")
  if cfg.engine not in ["text-davinci-003", "text-curie-001", "text-babbage-001", "text-ada-001"]:
    raise ValueError("The OpenAI model is not supported now.")

  model        = OpenAI(cfg.api_key, cfg.engine) 
  nlp_prompter = Prompter(model)

  if not cfg.zero_shot:
    data = json.load(open(cfg.data_path,'r'))
  else:
    data = []
  task = {
    "ner_cn": "ner_cn.jinja",
    "ner_en": "ner_en.jinja",
    "rte_cn": "rte_cn.jinja",
    "rte_en": "rte_en.jinja",
    "ee_cn": "ee_cn.jinja",
    "ee_en": "ee_en.jinja",
  }
  if cfg.task not in task:
    raise ValueError(f"The task is not supported now.")
  result = nlp_prompter.fit(task[cfg.task], text_input = text, examples = data, domain = cfg.domain, labels = None)
                    
  logger.info(result['text'])

if __name__ == '__main__':
  main()
