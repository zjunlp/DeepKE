from tensorboardX import SummaryWriter

class Summarizer(object):
    def __init__(self, logdir='./log'):
        self.writer = SummaryWriter(logdir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def text_summary(self, tag, value, step):
        self.writer.add_text(tag, value, step)

def generate_vocabs(datasets):
    event_type_set = set()
    role_type_set = set()
    for dataset in datasets:
        event_type_set.update(dataset.event_type_set)
        role_type_set.update(dataset.role_type_set)
    
    event_type_itos = sorted(event_type_set)
    role_type_itos = sorted(role_type_set)
    
    event_type_stoi = {k: i for i, k in enumerate(event_type_itos)}
    role_type_stoi = {k: i for i, k in enumerate(role_type_itos)}
    
    return {
        'event_type_itos': event_type_itos,
        'event_type_stoi': event_type_stoi,
        'role_type_itos': role_type_itos,
        'role_type_stoi': role_type_stoi,
    }

def safe_div(num, denom):
    if denom > 0:
        return num / denom
    else:
        return 0

def compute_f1(predicted, gold, matched):
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1
