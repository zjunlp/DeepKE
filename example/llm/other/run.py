

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

  text = cfg.text
  
  model        = OpenAI(cfg.api_key, cfg.engine) 
  nlp_prompter = Prompter(model)
  
  if not cfg.zero_shot:
    data_path = os.path.join(os.path.join(cfg.cwd,'data'),cfg.task +'.jinja')
    data = json.load(open(data_path,'r'))
  else:
    data = []
  task = {
    "ner": "ner.jinja",
    "rte": "rte.jinja",
    "ee": "ee.jinja",
  }
  result = nlp_prompter.fit(task[cfg.task], text_input = text, examples = data, domain = cfg.domain, labels = None)
                    
  logger.info(result['text'])

if __name__ == '__main__':
  main()