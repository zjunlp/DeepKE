import os, sys, json, logging, pprint, tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from deepke.event_extraction.standard.degree.model import GenerativeModel
from deepke.event_extraction.standard.degree.data import GenDataset, EEDataset
from deepke.event_extraction.standard.degree.utils import compute_f1
from deepke.event_extraction.standard.degree.template_generate_ace import *
import ipdb
import hydra
from hydra import utils
from omegaconf import OmegaConf, open_dict

# logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)


def get_span_idx(pieces, token_start_idxs, span, tokenizer, trigger_span=None):
    """
    This function is how we map the generated prediction back to span prediction.

    Detailed Explanation:
        We will first split our prediction and use tokenizer to tokenize our predicted "span" into pieces. Then, we will find whether we can find a continuous span in the original "pieces" can match tokenized "span".

    If it is an argument/relation extraction task, we will return the one which is closest to the trigger_span.
    """
    words = []
    for s in span.split(' '):
        words.extend(tokenizer.encode(s, add_special_tokens=True)[1:-1])  # ignore [SOS] and [EOS]

    candidates = []
    for i in range(len(pieces)):
        j = 0
        k = 0
        while j < len(words) and i + k < len(pieces):
            if pieces[i + k] == words[j]:
                j += 1
                k += 1
            elif tokenizer.decode(words[j]) == "":
                j += 1
            elif tokenizer.decode(pieces[i + k]) == "":
                k += 1
            else:
                break
        if j == len(words):
            candidates.append((i, i + k))

    candidates = [(token_start_idxs.index(c1), token_start_idxs.index(c2)) for c1, c2 in candidates if
                  c1 in token_start_idxs and c2 in token_start_idxs]
    if len(candidates) < 1:
        return -1, -1
    else:
        if trigger_span is None:
            return candidates[0]
        else:
            return sorted(candidates, key=lambda x: np.abs(trigger_span[0] - x[0]))[0]


def get_span_idx_tri(pieces, token_start_idxs, span, tokenizer, trigger_span=None):
    """
    This function is how we map the generated prediction back to span prediction.

    Detailed Explanation:
        We will first split our prediction and use tokenizer to tokenize our predicted "span" into pieces. Then, we will find whether we can find a continuous span in the original "pieces" can match tokenized "span".

    If it is an argument/relation extraction task, we will return the one which is closest to the trigger_span.
    """
    words = []
    for s in span.split(' '):
        words.extend(tokenizer.encode(s, add_special_tokens=True)[1:-1])  # ignore [SOS] and [EOS]

    candidates = []
    for i in range(len(pieces)):
        j = 0
        k = 0
        while j < len(words) and i + k < len(pieces):
            if pieces[i + k] == words[j]:
                j += 1
                k += 1
            elif tokenizer.decode(words[j]) == "":
                j += 1
            elif tokenizer.decode(pieces[i + k]) == "":
                k += 1
            else:
                break
        if j == len(words):
            candidates.append((i, i + k))

    candidates = [(token_start_idxs.index(c1), token_start_idxs.index(c2)) for c1, c2 in candidates if
                  c1 in token_start_idxs and c2 in token_start_idxs]
    if len(candidates) < 1:
        return [(-1, -1)]
    else:
        if trigger_span is None:
            return candidates
        else:
            return sorted(candidates, key=lambda x: np.abs(trigger_span[0] - x[0]))


def cal_scores(gold_triggers, pred_triggers, gold_roles, pred_roles):
    # tri_id
    gold_tri_id_num, pred_tri_id_num, match_tri_id_num = 0, 0, 0
    for gold_trigger, pred_trigger in zip(gold_triggers, pred_triggers):
        gold_set = set([(t[0], t[1]) for t in gold_trigger])
        pred_set = set([(t[0], t[1]) for t in pred_trigger])
        gold_tri_id_num += len(gold_set)
        pred_tri_id_num += len(pred_set)
        match_tri_id_num += len(gold_set & pred_set)

    # tri_cls
    gold_tri_cls_num, pred_tri_cls_num, match_tri_cls_num = 0, 0, 0
    for gold_trigger, pred_trigger in zip(gold_triggers, pred_triggers):
        gold_set = set(gold_trigger)
        pred_set = set(pred_trigger)
        gold_tri_cls_num += len(gold_set)
        pred_tri_cls_num += len(pred_set)
        match_tri_cls_num += len(gold_set & pred_set)

    # arg_id
    gold_arg_id_num, pred_arg_id_num, match_arg_id_num = 0, 0, 0
    for gold_role, pred_role in zip(gold_roles, pred_roles):
        gold_set = set([(r[0][2],) + r[1][:-1] for r in gold_role])
        pred_set = set([(r[0][2],) + r[1][:-1] for r in pred_role])

        gold_arg_id_num += len(gold_set)
        pred_arg_id_num += len(pred_set)
        match_arg_id_num += len(gold_set & pred_set)

    # arg_cls
    gold_arg_cls_num, pred_arg_cls_num, match_arg_cls_num = 0, 0, 0
    for gold_role, pred_role in zip(gold_roles, pred_roles):
        gold_set = set([(r[0][2],) + r[1] for r in gold_role])
        pred_set = set([(r[0][2],) + r[1] for r in pred_role])

        gold_arg_cls_num += len(gold_set)
        pred_arg_cls_num += len(pred_set)
        match_arg_cls_num += len(gold_set & pred_set)

    scores = {
        'tri_id': (gold_tri_id_num, pred_tri_id_num, match_tri_id_num) + compute_f1(gold_tri_id_num, pred_tri_id_num,
                                                                                    match_tri_id_num),
        'tri_cls': (gold_tri_cls_num, pred_tri_cls_num, match_tri_cls_num) + compute_f1(gold_tri_cls_num,
                                                                                        pred_tri_cls_num,
                                                                                        match_tri_cls_num),
        'arg_id': (gold_arg_id_num, pred_arg_id_num, match_arg_id_num) + compute_f1(gold_arg_id_num, pred_arg_id_num,
                                                                                    match_arg_id_num),
        'arg_cls': (gold_arg_cls_num, pred_arg_cls_num, match_arg_cls_num) + compute_f1(gold_arg_cls_num,
                                                                                        pred_arg_cls_num,
                                                                                        match_arg_cls_num),
    }

    return scores


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
    config.vocab_file = os.path.join(config.cwd, config.vocab_file)
    config.e2e_model = os.path.join(config.cwd, config.e2e_model)

    template_file = "deepke.event_extraction.standard.degree.template_generate_ace"

    # fix random seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.enabled = False

    # set GPU device
    torch.cuda.set_device(config.gpu_device)

    # check valid styles
    assert np.all([style in ['event_type_sent', 'keywords', 'template'] for style in config.input_style])
    assert np.all([style in ['trigger:sentence', 'argument:sentence'] for style in config.output_style])

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=config.cache_dir)
    special_tokens = ['<Trigger>', '<sep>']
    tokenizer.add_tokens(special_tokens)

    # if args.eval_batch_size:
    #     config.eval_batch_size = args.eval_batch_size

    # load data
    dev_set = EEDataset(tokenizer, config.dev_file, max_length=config.max_length)
    test_set = EEDataset(tokenizer, config.test_file, max_length=config.max_length)
    dev_batch_num = len(dev_set) // config.eval_batch_size + (len(dev_set) % config.eval_batch_size != 0)
    test_batch_num = len(test_set) // config.eval_batch_size + (len(test_set) % config.eval_batch_size != 0)
    with open(config.vocab_file) as f:
        vocab = json.load(f)

    # load model
    logger.info(f"Loading model from {config.e2e_model}")
    model = GenerativeModel(config, tokenizer)
    model.load_state_dict(torch.load(config.e2e_model, map_location=f'cuda:{config.gpu_device}'))
    model.cuda(device=config.gpu_device)
    model.eval()

    # eval dev set
    if not config.no_dev:
        progress = tqdm.tqdm(total=dev_batch_num, ncols=75, desc='Dev')
        dev_gold_triggers, dev_gold_roles, dev_pred_triggers, dev_pred_roles = [], [], [], []

        for batch in DataLoader(dev_set, batch_size=config.eval_batch_size, shuffle=False, collate_fn=dev_set.collate_fn):
            progress.update(1)

            p_triggers = [[] for _ in range(config.eval_batch_size)]
            p_roles = [[] for _ in range(config.eval_batch_size)]
            for event_type in vocab['event_type_itos']:
                theclass = getattr(sys.modules[template_file], event_type.replace(':', '_').replace('-', '_'), False)

                inputs = []
                for tokens in batch.tokens:
                    template = theclass(config.input_style, config.output_style, tokens, event_type)
                    inputs.append(template.generate_input_str(''))

                inputs = tokenizer(inputs, return_tensors='pt', padding=True, max_length=config.max_length)
                enc_idxs = inputs['input_ids'].cuda()
                enc_attn = inputs['attention_mask'].cuda()

                outputs = model.model.generate(input_ids=enc_idxs, attention_mask=enc_attn, num_beams=4,
                                               max_length=config.max_output_length)
                final_outputs = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for
                                 output in outputs]

                for bid, (tokens, p_text) in enumerate(zip(batch.tokens, final_outputs)):
                    template = theclass(config.input_style, config.output_style, tokens, event_type)
                    pred_object = template.decode(p_text)

                    pred_trigger_object = []
                    pred_argument_object = []
                    for obj in pred_object:
                        if obj[1] == event_type:
                            pred_trigger_object.append(obj)
                        else:
                            pred_argument_object.append(obj)

                    # decode triggers
                    triggers_ = [mention + (event_type, kwargs) for span, _, kwargs in pred_trigger_object for mention in
                                 get_span_idx_tri(batch.piece_idxs[bid], batch.token_start_idxs[bid], span, tokenizer)]
                    triggers_ = [t for t in triggers_ if t[0] != -1]
                    p_triggers_ = [t[:-1] for t in triggers_]
                    p_triggers_ = list(set(p_triggers_))
                    p_triggers[bid].extend(p_triggers_)

                    # decode arguments
                    tri_id2obj = {}
                    for t in triggers_:
                        try:
                            tri_id2obj[t[3]['tri counter']] = (t[0], t[1], t[2])
                        except:
                            ipdb.set_trace()

                    roles_ = []
                    for span, role_type, kwargs in pred_argument_object:
                        corres_tri_id = kwargs['cor tri cnt']
                        if corres_tri_id in tri_id2obj.keys():
                            arg_span = get_span_idx(batch.piece_idxs[bid], batch.token_start_idxs[bid], span, tokenizer,
                                                    tri_id2obj[corres_tri_id])
                            if arg_span[0] != -1:
                                roles_.append((tri_id2obj[corres_tri_id], (arg_span[0], arg_span[1], role_type)))
                        else:
                            arg_span = get_span_idx(batch.piece_idxs[bid], batch.token_start_idxs[bid], span, tokenizer)
                            if arg_span[0] != -1:
                                roles_.append(((0, 1, event_type), (arg_span[0], arg_span[1], role_type)))

                    p_roles[bid].extend(roles_)

            p_roles = [list(set(role)) for role in p_roles]

            if config.ignore_first_header:
                for bid, wnd_id in enumerate(batch.wnd_ids):
                    if int(wnd_id.split('-')[-1]) < 4:
                        p_triggers[bid] = []
                        p_roles[bid] = []

            dev_gold_triggers.extend(batch.triggers)
            dev_gold_roles.extend(batch.roles)
            dev_pred_triggers.extend(p_triggers)
            dev_pred_roles.extend(p_roles)

        progress.close()

        # calculate scores
        dev_scores = cal_scores(dev_gold_triggers, dev_pred_triggers, dev_gold_roles, dev_pred_roles)

        print("---------------------------------------------------------------------")
        print('Trigger I  - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
            dev_scores['tri_id'][3] * 100.0, dev_scores['tri_id'][2], dev_scores['tri_id'][1],
            dev_scores['tri_id'][4] * 100.0, dev_scores['tri_id'][2], dev_scores['tri_id'][0],
            dev_scores['tri_id'][5] * 100.0))
        print('Trigger C  - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
            dev_scores['tri_cls'][3] * 100.0, dev_scores['tri_cls'][2], dev_scores['tri_cls'][1],
            dev_scores['tri_cls'][4] * 100.0, dev_scores['tri_cls'][2], dev_scores['tri_cls'][0],
            dev_scores['tri_cls'][5] * 100.0))
        print("---------------------------------------------------------------------")
        print('Role I     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
            dev_scores['arg_id'][3] * 100.0, dev_scores['arg_id'][2], dev_scores['arg_id'][1],
            dev_scores['arg_id'][4] * 100.0, dev_scores['arg_id'][2], dev_scores['arg_id'][0],
            dev_scores['arg_id'][5] * 100.0))
        print('Role C     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
            dev_scores['arg_cls'][3] * 100.0, dev_scores['arg_cls'][2], dev_scores['arg_cls'][1],
            dev_scores['arg_cls'][4] * 100.0, dev_scores['arg_cls'][2], dev_scores['arg_cls'][0],
            dev_scores['arg_cls'][5] * 100.0))
        print("---------------------------------------------------------------------")

    # test set
    progress = tqdm.tqdm(total=test_batch_num, ncols=75, desc='Test')
    test_gold_triggers, test_gold_roles, test_pred_triggers, test_pred_roles = [], [], [], []
    write_object = []
    for batch in DataLoader(test_set, batch_size=config.eval_batch_size, shuffle=False, collate_fn=test_set.collate_fn):
        progress.update(1)
        p_triggers = [[] for _ in range(config.eval_batch_size)]
        p_roles = [[] for _ in range(config.eval_batch_size)]
        p_texts = [[] for _ in range(config.eval_batch_size)]
        for event_type in vocab['event_type_itos']:
            theclass = getattr(sys.modules[template_file], event_type.replace(':', '_').replace('-', '_'), False)

            inputs = []
            for tokens in batch.tokens:
                template = theclass(config.input_style, config.output_style, tokens, event_type)
                inputs.append(template.generate_input_str(''))
            # for tokens in batch.tokens:
            #     template = theclass(config.input_style, config.output_style, tokens, event_type)
            #     inputs.append(template.generate_input_str(''))
            inputs = tokenizer(inputs, return_tensors='pt', padding=True, max_length=config.max_length)
            enc_idxs = inputs['input_ids'].cuda()
            enc_attn = inputs['attention_mask'].cuda()

            outputs = model.model.generate(input_ids=enc_idxs, attention_mask=enc_attn, num_beams=4,
                                           max_length=config.max_output_length)
            final_outputs = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for
                             output in outputs]

            for bid, (tokens, p_text) in enumerate(zip(batch.tokens, final_outputs)):
                template = theclass(config.input_style, config.output_style, tokens, event_type)
                pred_object = template.decode(p_text)

                pred_trigger_object = []
                pred_argument_object = []
                for obj in pred_object:
                    if obj[1] == event_type:
                        pred_trigger_object.append(obj)
                    else:
                        pred_argument_object.append(obj)

                # decode triggers
                triggers_ = [mention + (event_type, kwargs) for span, _, kwargs in pred_trigger_object for mention in
                             get_span_idx_tri(batch.piece_idxs[bid], batch.token_start_idxs[bid], span, tokenizer)]

                triggers_ = [t for t in triggers_ if t[0] != -1]
                p_triggers_ = [t[:-1] for t in triggers_]
                p_triggers_ = list(set(p_triggers_))

                p_triggers[bid].extend(p_triggers_)

                # decode arguments
                tri_id2obj = {}
                for t in triggers_:
                    tri_id2obj[t[3]['tri counter']] = (t[0], t[1], t[2])

                roles_ = []
                for span, role_type, kwargs in pred_argument_object:
                    corres_tri_id = kwargs['cor tri cnt']
                    if corres_tri_id in tri_id2obj.keys():
                        arg_span = get_span_idx(batch.piece_idxs[bid], batch.token_start_idxs[bid], span, tokenizer,
                                                tri_id2obj[corres_tri_id])
                        if arg_span[0] != -1:
                            roles_.append((tri_id2obj[corres_tri_id], (arg_span[0], arg_span[1], role_type)))
                    else:
                        arg_span = get_span_idx(batch.piece_idxs[bid], batch.token_start_idxs[bid], span, tokenizer)
                        if arg_span[0] != -1:
                            roles_.append(((0, 1, event_type), (arg_span[0], arg_span[1], role_type)))

                p_roles[bid].extend(roles_)
                p_texts[bid].append(p_text)

        p_roles = [list(set(role)) for role in p_roles]

        if config.ignore_first_header:
            for bid, wnd_id in enumerate(batch.wnd_ids):
                if int(wnd_id.split('-')[-1]) < 4:
                    p_triggers[bid] = []
                    p_roles[bid] = []

        test_gold_triggers.extend(batch.triggers)
        test_gold_roles.extend(batch.roles)
        test_pred_triggers.extend(p_triggers)
        test_pred_roles.extend(p_roles)
        for gt, gr, pt, pr, te in zip(batch.triggers, batch.roles, p_triggers, p_roles, p_texts):
            write_object.append({
                "pred text": te,
                "pred triggers": pt,
                "gold triggers": gt,
                "pred roles": pr,
                "gold roles": gr
            })

    progress.close()

    # calculate scores
    dev_scores = cal_scores(test_gold_triggers, test_pred_triggers, test_gold_roles, test_pred_roles)

    print("---------------------------------------------------------------------")
    print('Trigger I  - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        dev_scores['tri_id'][3] * 100.0, dev_scores['tri_id'][2], dev_scores['tri_id'][1],
        dev_scores['tri_id'][4] * 100.0, dev_scores['tri_id'][2], dev_scores['tri_id'][0], dev_scores['tri_id'][5] * 100.0))
    print('Trigger C  - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        dev_scores['tri_cls'][3] * 100.0, dev_scores['tri_cls'][2], dev_scores['tri_cls'][1],
        dev_scores['tri_cls'][4] * 100.0, dev_scores['tri_cls'][2], dev_scores['tri_cls'][0],
        dev_scores['tri_cls'][5] * 100.0))
    print("---------------------------------------------------------------------")
    print('Role I     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        dev_scores['arg_id'][3] * 100.0, dev_scores['arg_id'][2], dev_scores['arg_id'][1],
        dev_scores['arg_id'][4] * 100.0, dev_scores['arg_id'][2], dev_scores['arg_id'][0], dev_scores['arg_id'][5] * 100.0))
    print('Role C     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        dev_scores['arg_cls'][3] * 100.0, dev_scores['arg_cls'][2], dev_scores['arg_cls'][1],
        dev_scores['arg_cls'][4] * 100.0, dev_scores['arg_cls'][2], dev_scores['arg_cls'][0],
        dev_scores['arg_cls'][5] * 100.0))
    print("---------------------------------------------------------------------")

    if config.write_file:
        with open(config.write_file, 'w') as fw:
            json.dump(write_object, fw, indent=4)

if __name__ == '__main__':
    main()