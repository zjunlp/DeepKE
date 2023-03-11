# /usr/bin/env python
# coding=utf-8
"""train with valid"""
import os
import sys
import argparse
import hydra
from hydra.utils import get_original_cwd
import torch
import logging
import tqdm
import wandb
from models.const import *
from models.data_structures import *
from models.entityModels import *
from models.entityUtils import *
from models.relationModels import *
from models.relationUtils import *
from run_entity import *
from run_relation import *


def entity_train(args):
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

    logger.info(sys.argv)
    logger.info(args)
    
    ner_label2id, ner_id2label = get_labelmap(task_ner_labels[args.task])
    
    num_ner_labels = len(task_ner_labels[args.task]) + 1
    model = EntityModel(args, num_ner_labels=num_ner_labels)

    # print(args.dev_data)
    dev_data = Dataset(args.dev_data)
    dev_samples, dev_ner = convert_dataset_to_samples(dev_data, args.entity_max_span_length, ner_label2id=ner_label2id, context_window=args.entity_context_window)
    dev_batches = batchify(dev_samples, args.entity_eval_batch_size)

    if args.entity_do_train:
        train_data = Dataset(args.train_data)
        train_samples, train_ner = convert_dataset_to_samples(train_data, args.entity_max_span_length, ner_label2id=ner_label2id, context_window=args.entity_context_window)
        train_batches = batchify(train_samples, args.entity_train_batch_size)
        best_result = 0.0

        param_optimizer = list(model.bert_model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                if 'bert' in n]},
            {'params': [p for n, p in param_optimizer
                if 'bert' not in n], 'lr': args.entity_task_learning_rate}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.entity_learning_rate, correct_bias=not(args.entity_bertadam))
        t_total = len(train_batches) * args.entity_num_epoch
        scheduler = get_linear_schedule_with_warmup(optimizer, int(t_total*args.entity_warmup_proportion), t_total)
        
        tr_loss = 0
        tr_examples = 0
        global_step = 0
        eval_step = len(train_batches) // args.entity_eval_per_epoch
        for _ in tqdm(range(args.entity_num_epoch)):
            if args.entity_train_shuffle:
                random.shuffle(train_batches)
            for i in tqdm(range(len(train_batches))):
                output_dict = model.run_batch(train_batches[i], training=True)
                loss = output_dict['ner_loss']
                loss.backward()

                tr_loss += loss.item()
                tr_examples += len(train_batches[i])
                global_step += 1

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if global_step % args.entity_print_loss_step == 0:
                    logger.info('Epoch=%d, iter=%d, loss=%.5f'%(_, i, tr_loss / tr_examples))
                    tr_loss = 0
                    tr_examples = 0

                if global_step % eval_step == 0:
                    f1 = entity_evaluate(model, dev_batches, dev_ner)
                    if f1 > best_result:
                        best_result = f1
                        logger.info('!!! Best valid (epoch=%d): %.2f' % (_, f1*100))
                        entity_save_model(model, args)

    if args.entity_do_eval:
        args.entity_bert_model_dir = args.entity_output_dir
        model = EntityModel(args, num_ner_labels=num_ner_labels)

        test_data = Dataset(args.dev_data)
        prediction_file = os.path.join(args.entity_output_dir, args.entity_dev_pred_filename)
        test_samples, test_ner = convert_dataset_to_samples(test_data, args.entity_max_span_length, ner_label2id=ner_label2id, context_window=args.entity_context_window)
        test_batches = batchify(test_samples, args.entity_eval_batch_size)
        entity_evaluate(model, test_batches, test_ner)
        entity_output_ner_predictions(model, test_batches, test_data, output_file=prediction_file, ner_id2label=ner_id2label)

        if args.entity_eval_test:
            test_data = Dataset(args.test_data)
            prediction_file = os.path.join(args.entity_output_dir, args.entity_test_pred_filename)
            test_samples, test_ner = convert_dataset_to_samples(test_data, args.entity_max_span_length, ner_label2id=ner_label2id, context_window=args.entity_context_window)
            test_batches = batchify(test_samples, args.entity_eval_batch_size)
            entity_evaluate(model, test_batches, test_ner)
            entity_output_ner_predictions(model, test_batches, test_data, output_file=prediction_file, ner_id2label=ner_id2label)

def relation_train(args):
    if 'albert' in args.relation_model:
        RelationModel = AlbertForRelation
        args.relation_add_new_tokens = True
    else:
        RelationModel = BertForRelation

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    # train set
    if args.relation_do_train:
        train_dataset, train_examples, train_nrel = generate_relation_data(args.relation_train_file, use_gold=True, context_window=args.relation_context_window)
    # dev set
    if (args.relation_do_eval and args.relation_do_train) or (args.relation_do_eval and not(args.relation_eval_test)):
        eval_dataset, eval_examples, eval_nrel = generate_relation_data(os.path.join(args.entity_output_dir, args.relation_entity_predictions_dev), use_gold=args.relation_eval_with_gold, context_window=args.relation_context_window)
    # test set
    if args.relation_eval_test:
        test_dataset, test_examples, test_nrel = generate_relation_data(os.path.join(args.entity_output_dir, args.relation_entity_predictions_test), use_gold=args.relation_eval_with_gold, context_window=args.relation_context_window)

    relation_setseed(args,args.relation_seed)

    if not args.relation_do_train and not args.relation_do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not os.path.exists(args.relation_output_dir):
        os.makedirs(args.relation_output_dir)
    if args.relation_do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.relation_output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.relation_output_dir, "eval.log"), 'w'))
    logger.info(sys.argv)
    logger.info(args)
    logger.info("device: {}, n_gpu: {}".format(
        device, n_gpu))

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

    if args.relation_do_eval and (args.relation_do_train or not(args.relation_eval_test)):
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

    if args.relation_do_train:
        train_features = relation_convert_examples_to_features(
            train_examples, label2id, args.relation_max_seq_length, tokenizer, special_tokens, unused_tokens=not(args.relation_add_new_tokens))
        if args.relation_train_mode == 'sorted' or args.relation_train_mode == 'random_sorted':
            train_features = sorted(train_features, key=lambda f: np.sum(f.input_mask))
        else:
            random.shuffle(train_features)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_sub_idx = torch.tensor([f.sub_idx for f in train_features], dtype=torch.long)
        all_obj_idx = torch.tensor([f.obj_idx for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_sub_idx, all_obj_idx)
        train_dataloader = DataLoader(train_data, batch_size=args.relation_train_batch_size)
        train_batches = [batch for batch in train_dataloader]

        num_train_optimization_steps = len(train_dataloader) * args.relation_num_train_epochs

        logger.info("***** Training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.relation_train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        best_result = None
        eval_step = max(1, len(train_batches) // args.relation_eval_per_epoch)
        
        lr = args.relation_learning_rate
        model = RelationModel.from_pretrained(
            args.relation_model, cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE), num_rel_labels=num_labels)
        if hasattr(model, 'bert'):
            model.bert.resize_token_embeddings(len(tokenizer))
        elif hasattr(model, 'albert'):
            model.albert.resize_token_embeddings(len(tokenizer))
        else:
            raise TypeError("Unknown model class")

        model.to(device)
        if n_gpu > 1 and args.relation_single_card == False:
            model = torch.nn.DataParallel(model)

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=not(args.relation_bertadam), no_deprecation_warning=True)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(num_train_optimization_steps * args.relation_warmup_proportion), num_train_optimization_steps)

        start_time = time.time()
        global_step = 0
        tr_loss = 0
        nb_tr_examples = 0
        nb_tr_steps = 0
        for epoch in range(int(args.relation_num_train_epochs)):
            model.train()
            logger.info("Start epoch #{} (lr = {})...".format(epoch, lr))
            if args.relation_train_mode == 'random' or args.relation_train_mode == 'random_sorted':
                random.shuffle(train_batches)
            for step, batch in enumerate(train_batches):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, sub_idx, obj_idx = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids, sub_idx, obj_idx)
                if n_gpu > 1:
                    loss = loss.mean()

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if (step + 1) % eval_step == 0:
                    logger.info('Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'.format(
                                epoch, step + 1, len(train_batches),
                                time.time() - start_time, tr_loss / nb_tr_steps))
                    save_model = False
                    if args.relation_do_eval:
                        preds, result, logits = relation_evaluate(model, device, eval_dataloader, eval_label_ids, num_labels, e2e_ngold=eval_nrel)
                        model.train()
                        result['global_step'] = global_step
                        result['epoch'] = epoch
                        result['learning_rate'] = lr
                        result['batch_size'] = args.relation_train_batch_size

                        if (best_result is None) or (result[args.eval_metric] > best_result[args.eval_metric]):
                            best_result = result
                            logger.info("!!! Best dev %s (lr=%s, epoch=%d): %.2f" %
                                        (args.eval_metric, str(lr), epoch, result[args.eval_metric] * 100.0))
                            relation_save_trained_model(args.relation_output_dir, model, tokenizer)

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

wandb.init(project="DeepKE_TRIPLE_PURE")
wandb.watch_called = False
@hydra.main(config_path="conf", config_name="config")
def main(args):
    cwd = get_original_cwd()
    os.chdir(cwd)
    print("Start training Entity Model")
    entity_train(args)
    print("Start training Relation Model")
    relation_train(args)

if __name__ == "__main__":
    main()