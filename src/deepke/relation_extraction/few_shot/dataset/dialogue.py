from .base_data_module import BaseDataModule
from .processor import get_dataset, processors
from transformers import AutoTokenizer



class REDataset(BaseDataModule):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.processor = processors[self.args.task_name](self.args.data_dir, self.args.use_prompt)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)
        
        
        use_gpt = "gpt" in args.model_name_or_path

        rel2id = self.processor.get_labels()
        self.num_labels = len(rel2id)

        entity_list = ["[object_start]", "[object_end]", "[subject_start]", "[subject_end]"]
        class_list = [f"[class{i}]" for i in range(1, self.num_labels+1)]

        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': entity_list})
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': class_list})
        if use_gpt:
            self.tokenizer.add_special_tokens({'cls_token': "[CLS]"})
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        so_list = ["[sub]", "[obj]"]
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': so_list})

        prompt_tokens = [f"[T{i}]" for i in range(1,6)]
        self.tokenizer.add_special_tokens({'additional_special_tokens': prompt_tokens})




    def setup(self, stage=None):
        self.data_train = get_dataset("train", self.args, self.tokenizer, self.processor)
        self.data_val = get_dataset("dev", self.args, self.tokenizer, self.processor)
        self.data_test = get_dataset("test", self.args, self.tokenizer, self.processor)


    def prepare_data(self):
        pass

    def get_tokenizer(self):
        return self.tokenizer

