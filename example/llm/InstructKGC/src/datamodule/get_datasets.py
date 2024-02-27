from datasets import load_dataset
from datamodule.template import DatasetAttr
from datamodule.preprocess import preprocess_dataset



def rename_dataset_colume(dataset, dataset_attr, column_names):
    for column_name in ["prompt", "query", "response", "history"]: 
        original_column = getattr(dataset_attr, column_name)
        if getattr(dataset_attr, column_name) and original_column in column_names and original_column != column_name:
            dataset = dataset.rename_column(getattr(dataset_attr, column_name), column_name)
    return dataset


def load_train_datasets(training_args, data_args):
    train_data_dict = None
    dataset_attr = DatasetAttr()
    if data_args.train_file is not None:
        train_data = load_dataset(
            path="json",
            data_files=data_args.train_file,
        )["train"]
        column_names = list(next(iter(train_data)).keys())
        train_data = rename_dataset_colume(train_data, dataset_attr, column_names)
    if data_args.valid_file is not None:
        valid_data = load_dataset(
            path="json",
            data_files=data_args.valid_file,
        )["train"]
        column_names = list(next(iter(valid_data)).keys())
        valid_data = rename_dataset_colume(valid_data, dataset_attr, column_names)
    else:    
        if training_args.do_eval and data_args.val_set_size > 0:
            train_val = train_data.train_test_split(test_size=data_args.val_set_size, shuffle=True, seed=42)
            train_data = train_val["train"]
            valid_data = train_val["test"]
    return train_data_dict, train_data, valid_data



def process_datasets(training_args, data_args, finetuning_args, tokenizer, train_data_dict, train_data, valid_data):
    # 数据预处理
    if training_args.do_train:
        train_data = preprocess_dataset(train_data, tokenizer, data_args, training_args, stage=finetuning_args.stage)
    if training_args.do_eval:
        valid_data = preprocess_dataset(valid_data, tokenizer, data_args, training_args, stage=finetuning_args.stage)
    return train_data, valid_data

