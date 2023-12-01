import os
import json
import argparse
import hydra
from hydra.utils import get_original_cwd
import torch
import logging
import tqdm
import wandb
from models import *
from models.data_structures import Dataset, evaluate_predictions
from run_entity import *
from run_relation import *



def model_predict(args):
    if 'albert' in args.entity_model:
        logger.info('Use Albert: %s'%args.entity_model)
        args.entity_use_albert = True

    entity_setseed(args.entity_seed)

    if not os.path.exists(args.entity_output_dir):
        os.makedirs(args.entity_output_dir)

    if args.entity_do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.entity_output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.entity_output_dir, "eval.log"), 'w'))

    ner_label2id, ner_id2label = get_labelmap(task_ner_labels[args.task])
    
    num_ner_labels = len(task_ner_labels[args.task]) + 1
    model = EntityModel(args, num_ner_labels=num_ner_labels)

    # print(args.dev_data)
    dev_data = Dataset(args.dev_data)
    dev_samples, dev_ner = convert_dataset_to_samples(dev_data, args.entity_max_span_length, ner_label2id=ner_label2id, context_window=args.entity_context_window)
    dev_batches = batchify(dev_samples, args.entity_eval_batch_size)

    if args.entity_do_eval:
        args.entity_bert_model_dir = args.entity_output_dir
        model = EntityModel(args, num_ner_labels=num_ner_labels)

        if args.entity_eval_test:
            test_data = Dataset(args.test_data)
            prediction_file = os.path.join(args.entity_output_dir, args.entity_test_pred_filename)
        else:
            test_data = Dataset(args.dev_data)
            prediction_file = os.path.join(args.entity_output_dir, args.entity_dev_pred_filename)
        test_samples, test_ner = convert_dataset_to_samples(test_data, args.entity_max_span_length, ner_label2id=ner_label2id, context_window=args.entity_context_window)
        test_batches = batchify(test_samples, args.entity_eval_batch_size)
        entity_evaluate(model, test_batches, test_ner)
        entity_output_ner_predictions(model, test_batches, test_data, output_file=prediction_file, ner_id2label=ner_id2label)

    if 'albert' in args.relation_model:
        RelationModel = AlbertForRelation
        args.relation_add_new_tokens = True
    else:
        RelationModel = BertForRelation

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    # dev set
    if args.relation_do_eval and not(args.relation_eval_test):
        eval_dataset, eval_examples, eval_nrel = generate_relation_data(os.path.join(args.entity_output_dir, args.relation_entity_predictions_dev), use_gold=args.relation_eval_with_gold, context_window=args.relation_context_window)
    # test set
    if args.relation_eval_test:
        test_dataset, test_examples, test_nrel = generate_relation_data(os.path.join(args.entity_output_dir, args.relation_entity_predictions_test), use_gold=args.relation_eval_with_gold, context_window=args.relation_context_window)

    relation_setseed(args,args.relation_seed)

    if not os.path.exists(args.relation_output_dir):
        os.makedirs(args.relation_output_dir)
    if args.relation_do_eval:
        logger.addHandler(logging.FileHandler(os.path.join(args.relation_output_dir, "eval.log"), 'w'))

    # get label_list
    if os.path.exists(os.path.join(args.relation_output_dir, 'label_list.json')):
        with open(os.path.join(args.relation_output_dir, 'label_list.json'), 'r') as f:
            label_list = json.load(f)
    else:
        label_list = [args.relation_negative_label] + task_rel_labels[args.task]
        with open(os.path.join(args.relation_output_dir, 'label_list.json'), 'w') as f:
            json.dump(label_list, f)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    tokenizer = AutoTokenizer.from_pretrained(args.relation_model, do_lower_case=args.relation_do_lower_case)
    if args.relation_add_new_tokens:
        relation_add_marker_tokens(tokenizer, task_ner_labels[args.task])

    if os.path.exists(os.path.join(args.relation_output_dir, 'special_tokens.json')):
        with open(os.path.join(args.relation_output_dir, 'special_tokens.json'), 'r') as f:
            special_tokens = json.load(f)
    else:
        special_tokens = {}

    if args.relation_do_eval and not(args.relation_eval_test):
        eval_features = relation_convert_examples_to_features(
            eval_examples, label2id, args.relation_max_seq_length, tokenizer, special_tokens, unused_tokens=not(args.relation_add_new_tokens))
        logger.info("***** Dev *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.relation_eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_sub_idx = torch.tensor([f.sub_idx for f in eval_features], dtype=torch.long)
        all_obj_idx = torch.tensor([f.obj_idx for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_sub_idx, all_obj_idx)
        eval_dataloader = DataLoader(eval_data, batch_size=args.relation_eval_batch_size)
        eval_label_ids = all_label_ids
    with open(os.path.join(args.relation_output_dir, 'special_tokens.json'), 'w') as f:
        json.dump(special_tokens, f)

    evaluation_results = {}
    if args.relation_do_eval:
        logger.info(special_tokens)
        if args.relation_eval_test:
            eval_dataset = test_dataset
            eval_examples = test_examples
            eval_features = relation_convert_examples_to_features(
                test_examples, label2id, args.relation_max_seq_length, tokenizer, special_tokens, unused_tokens=not(args.relation_add_new_tokens))
            eval_nrel = test_nrel
            logger.info(special_tokens)
            logger.info("***** Test *****")
            logger.info("  Num examples = %d", len(test_examples))
            logger.info("  Batch size = %d", args.relation_eval_batch_size)
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
            all_sub_idx = torch.tensor([f.sub_idx for f in eval_features], dtype=torch.long)
            all_obj_idx = torch.tensor([f.obj_idx for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_sub_idx, all_obj_idx)
            eval_dataloader = DataLoader(eval_data, batch_size=args.relation_eval_batch_size)
            eval_label_ids = all_label_ids
        model = RelationModel.from_pretrained(args.relation_output_dir, num_rel_labels=num_labels)
        model.to(device)
        preds, result, logits = relation_evaluate(model, device, eval_dataloader, eval_label_ids, num_labels, e2e_ngold=eval_nrel)

        logger.info('*** Evaluation Results ***')
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

        relation_print_pred_json(eval_dataset, eval_examples, preds, id2label, os.path.join(args.relation_output_dir, args.relation_prediction_file))


def model_evaluate(args):
    prediction_file = os.path.join(args.relation_output_dir, args.relation_prediction_file)
    data = Dataset(prediction_file)
    eval_result = evaluate_predictions(data)
    print('Evaluation result %s'%(prediction_file))
    print('NER - P: %f, R: %f, F1: %f'%(eval_result['ner']['precision'], eval_result['ner']['recall'], eval_result['ner']['f1']))
    print('REL - P: %f, R: %f, F1: %f'%(eval_result['relation']['precision'], eval_result['relation']['recall'], eval_result['relation']['f1']))
    print('REL (strict) - P: %f, R: %f, F1: %f'%(eval_result['strict_relation']['precision'], eval_result['strict_relation']['recall'], eval_result['strict_relation']['f1']))

@hydra.main(config_path="conf", config_name="config")
def main(args):
    cwd = get_original_cwd()
    os.chdir(cwd)
    print("Start predicting dataset")
    model_predict(args)
    print("Start evaluatint predictions")
    model_evaluate(args)

if __name__ == "__main__":
    main()
