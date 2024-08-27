import sys
import ipdb

INPUT_STYLE_SET = ['event_type', 'event_type_sent', 'keywords', 'triggers', 'template']
OUTPUT_STYLE_SET = ['trigger:sentence', 'argument:sentence']
ROLE_PH_MAP = {
    'Person': 'somebody',
    'Entity': 'some people or some organization',
    'Defendant': 'somebody',
    'Prosecutor': 'some other',
    'Plaintiff': 'some other',
    'Buyer': 'someone',
    'Artifact': 'something',
    'Seller': 'some seller',
    'Destination': 'somewhere',
    'Origin': 'some place',
    'Vehicle': 'some vehicle',
    'Agent': 'somebody or some organization',
    'Attacker': 'some attacker',
    'Target': 'some facility, someone, or some organization',
    'Victim': 'some victim',
    'Instrument': 'some way',
    'Giver': 'someone',
    'Recipient': 'some other',
    'Org': 'some organization',
    'Place': 'somewhere',
    'Adjudicator': 'some adjudicator'
}

class eve_template_generator():
    def __init__(self, passage, triggers, roles, input_style, output_style, vocab, instance_base=False):
        """
        generate strctured information for events
        
        args:
            passage(List): a list of tokens
            triggers(List): a list of triggers
            roles(List): a list of Roles
            input_style(List): List of elements; elements belongs to INPUT_STYLE_SET
            input_style(List): List of elements; elements belongs to OUTPUT_STYLE_SET
            instance_base(Bool): if instance_base, we generate only one pair (use for trigger generation), else, we generate trigger_base (use for argument generation)
        """
        self.raw_passage = passage
        self.triggers = triggers
        self.roles = roles
        self.events = self.process_events(passage, triggers, roles)
        self.input_style = input_style
        self.output_style = output_style
        self.vocab = vocab
        self.event_templates = []
        if instance_base:
            for e_type in self.vocab['event_type_itos']:
                theclass = getattr(sys.modules[__name__], e_type.replace(':', '_').replace('-', '_'), False)
                if theclass:
                    self.event_templates.append(theclass(self.input_style, self.output_style, passage, e_type, self.events))
                else:
                    print(e_type)

        else:
            for event in self.events:
                theclass = getattr(sys.modules[__name__], event['event type'].replace(':', '_').replace('-', '_'), False)
                assert theclass
                self.event_templates.append(theclass(self.input_style, self.output_style, event['tokens'], event['event type'], event))
        self.data = [x.generate_pair(x.trigger_text) for x in self.event_templates]
        self.data = [x for x in self.data if x]

    def get_training_data(self):
        return self.data

    def process_events(self, passage, triggers, roles):
        """
        Given a list of token and event annotation, return a list of structured event

        structured_event:
        {
            'trigger text': str,
            'trigger span': (start, end),
            'event type': EVENT_TYPE(str),
            'arguments':{
                ROLE_TYPE(str):[{
                    'argument text': str,
                    'argument span': (start, end)
                }],
                ROLE_TYPE(str):...,
                ROLE_TYPE(str):....
            }
            'passage': PASSAGE
        }
        """
        
        events = {trigger: [] for trigger in triggers}

        for argument in roles:
            trigger = argument[0]
            events[trigger].append(argument)
        
        event_structures = []
        for trigger, arguments in events.items():
            eve_type = trigger[2]
            eve_text = ' '.join(passage[trigger[0]:trigger[1]])
            eve_span = (trigger[0], trigger[1])
            argus = {}
            for argument in arguments:
                role_type = argument[1][2]
                if role_type not in argus.keys():
                    argus[role_type] = []
                argus[role_type].append({
                    'argument text': ' '.join(passage[argument[1][0]:argument[1][1]]),
                    'argument span': (argument[1][0], argument[1][1]),
                })
            event_structures.append({
                'trigger text': eve_text,
                'trigger span': eve_span,
                'event type': eve_type,
                'arguments': argus,
                'passage': ' '.join(passage),
                'tokens': passage
            })
        return event_structures

class event_template():
    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        self.input_style = input_style
        self.output_style = output_style
        self.output_template = self.get_output_template()
        self.passage = ' '.join(passage)
        self.tokens = passage
        self.event_type = event_type
        if gold_event is not None:
            self.gold_event = gold_event
            if isinstance(gold_event, list):
                # instance base
                self.trigger_text = " and ".join([x['trigger text'] for x in gold_event if x['event type']==event_type])
                self.trigger_span = [x['trigger span'] for x in gold_event if x['event type']==event_type]
                self.arguments = [x['arguments'] for x in gold_event if x['event type']==event_type]
            else:
                # trigger base
                self.trigger_text = gold_event['trigger text']
                self.trigger_span = [gold_event['trigger span']]
                self.arguments = [gold_event['arguments']]         
        else:
            self.gold_event = None
        
    @classmethod
    def get_keywords(self):
        pass

    def generate_pair(self, query_trigger):
        """
        Generate model input sentence and output sentence pair
        """
        input_str = self.generate_input_str(query_trigger)
        output_str, gold_sample = self.generate_output_str(query_trigger)
        return (input_str, output_str, self.gold_event, gold_sample, self.event_type, self.tokens)

    def generate_input_str(self, query_trigger):
        return None

    def generate_output_str(self, query_trigger):
        return (None, False)

    def decode(self, prediction):
        pass

    def evaluate(self, predict_output):
        assert self.gold_event is not None
        # categorize prediction
        pred_trigger = []
        pred_argument = []
        for pred in predict_output:
            if pred[1] == self.event_type:
                pred_trigger.append(pred)
            else:
                pred_argument.append(pred)
        # trigger score
        gold_tri_num = len(self.trigger_span)
        pred_tris = []
        for pred in pred_trigger:
            pred_span = self.predstr2span(pred[0])
            if pred_span[0] > -1:
                pred_tris.append((pred_span[0], pred_span[1], pred[1]))
        pred_tri_num = len(pred_tris)
        match_tri = 0
        for pred in pred_tris:
            id_flag = False
            for gold_span in self.trigger_span:
                if gold_span[0] == pred[0] and gold_span[1] == pred[1]:
                    id_flag = True
            match_tri += int(id_flag)

        # argument score
        converted_gold = self.get_converted_gold()
        gold_arg_num = len(converted_gold)
        pred_arg = []
        for pred in pred_argument:
            # find corresponding trigger
            pred_span = None
            if isinstance(self.gold_event, list):
                # end2end case
                try:
                    # we need this ``try'' because we cannot gurantee the model will be bug-free on the matching
                    cor_tri = pred_trigger[pred[2]['cor tri cnt']]
                    cor_tri_span = self.predstr2span(cor_tri[0])[0]
                    if cor_tri_span > -1:
                        pred_span = self.predstr2span(pred[0], cor_tri_span)
                    else:
                        continue
                except Exception as e:
                    print(e)
            else:
                # argument only case
                pred_span = self.predstr2span(pred[0], self.trigger_span[0][0])
            if (pred_span is not None) and (pred_span[0] > -1):
                pred_arg.append((pred_span[0], pred_span[1], pred[1]))
        pred_arg = list(set(pred_arg))
        pred_arg_num = len(pred_arg)
        
        target = converted_gold
        match_id = 0
        match_type = 0
        for pred in pred_arg:
            id_flag = False
            id_type = False
            for gold in target:
                if gold[0]==pred[0] and gold[1]==pred[1]:
                    id_flag = True
                    if gold[2] == pred[2]:
                        id_type = True
                        break
            match_id += int(id_flag)
            match_type += int(id_type)
        return {
            'gold_tri_num': gold_tri_num, 
            'pred_tri_num': pred_tri_num,
            'match_tri_num': match_tri,
            'gold_arg_num': gold_arg_num,
            'pred_arg_num': pred_arg_num,
            'match_arg_id': match_id,
            'match_arg_cls': match_type
        }
    
    def get_converted_gold(self):
        converted_gold = []
        for argu in self.arguments:
            for arg_type, arg_list in argu.items():
                for arg in arg_list:
                    converted_gold.append((arg['argument span'][0], arg['argument span'][1], arg_type))
        return list(set(converted_gold))
    
    def predstr2span(self, pred_str, trigger_idx=None):
        sub_words = [_.strip() for _ in pred_str.strip().lower().split()]
        candidates=[]
        for i in range(len(self.tokens)):
            j = 0
            while j < len(sub_words) and i+j < len(self.tokens):
                if self.tokens[i+j].lower() == sub_words[j]:
                    j += 1
                else:
                    break
            if j == len(sub_words):
                candidates.append((i, i+len(sub_words)))
        if len(candidates) < 1:
            return -1, -1
        else:
            if trigger_idx is not None:
                return sorted(candidates, key=lambda x: abs(trigger_idx-x[0]))[0]
            else:
                return candidates[0]

class Life_Be_Born(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['born', 'birth', 'bore']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
                if o_style == 'argument:sentence':
                    output_template += ' \n {}'.format('somebody was born in somewhere.')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('life event, be-born sub-type')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('The event is related to life and someone is given birth to.')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
                if o_style == 'argument:sentence':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            " and ".join([ a['argument text'] for a in argu['Person']]) if "Person" in argu.keys() else ROLE_PH_MAP['Person'],
                            " and ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else ROLE_PH_MAP['Place']
                        )
                        output_texts.append("{} was born in {}.".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))

        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
                    if o_style == 'argument:sentence':
                        if used_o_cnt == cnt:
                            try:
                                for a_cnt, prediction in enumerate(full_pred.split(' <sep> ')):
                                    person = prediction.split(' was born in ', 1)[0]
                                    person = person.split(' and ')
                                    place = prediction.split(' was born in ', 1)[1].rsplit('.', 1)[0]
                                    place = place.split(' and ')
                                    for arg in person:
                                        if arg != ROLE_PH_MAP['Person']:
                                            output.append((arg, 'Person', {'cor tri cnt': a_cnt}))
                                    for arg in place:
                                        if arg != ROLE_PH_MAP['Place']:
                                            output.append((arg, 'Place', {'cor tri cnt': a_cnt}))
                            except:
                                pass
                        used_o_cnt += 1
                    
        return output

class Life_Marry(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['marry', 'marriage', 'married']
    
    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
                if o_style == 'argument:sentence':
                    output_template += ' \n {}'.format('somebody got married in somewhere.')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('life event, marry sub-type')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('The event is related to life and someone is married.')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
                if o_style == 'argument:sentence':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            " and ".join([ a['argument text'] for a in argu['Person']]) if "Person" in argu.keys() else ROLE_PH_MAP['Person'],
                            " and ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else ROLE_PH_MAP['Place']
                        )
                        output_texts.append("{} got married in {}.".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                    pass
                        used_o_cnt += 1
                    if o_style == 'argument:sentence':
                        if used_o_cnt == cnt:
                            try:
                                for a_cnt, prediction in enumerate(full_pred.split(' <sep> ')):
                                    person = prediction.split(' got married in ', 1)[0]
                                    person = person.split(' and ')
                                    place = prediction.split(' got married in ', 1)[1].rsplit('.', 1)[0]
                                    place = place.split(' and ')
                                    for arg in person:
                                        if arg != ROLE_PH_MAP['Person']:
                                            output.append((arg, 'Person', {'cor tri cnt': a_cnt}))
                                    for arg in place:
                                        if arg != ROLE_PH_MAP['Place']:
                                            output.append((arg, 'Place', {'cor tri cnt': a_cnt}))
                            except:
                                pass
                        used_o_cnt += 1
        return output

class Life_Divorce(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['divorce', 'divorced', 'Divorce']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
                if o_style == 'argument:sentence':
                    output_template += ' \n {}'.format('somebody divorced in somewhere.')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('life event, divorce sub-type')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('The event is related to life and someone was divorced.')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
                if o_style == 'argument:sentence':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            " and ".join([ a['argument text'] for a in argu['Person']]) if "Person" in argu.keys() else ROLE_PH_MAP['Person'],
                            " and ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else ROLE_PH_MAP['Place']
                        )
                        output_texts.append("{} divorced in {}.".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
                
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
                    if o_style == 'argument:sentence':
                        if used_o_cnt == cnt:
                            try:
                                for a_cnt, prediction in enumerate(full_pred.split(' <sep> ')):
                                    person = prediction.split(' divorced in ', 1)[0]
                                    person = person.split(' and ')
                                    place = prediction.split(' divorced in ', 1)[1].rsplit('.', 1)[0]
                                    place = place.split(' and ')
                                    for arg in person:
                                        if arg != ROLE_PH_MAP['Person']:
                                            output.append((arg, 'Person', {'cor tri cnt': a_cnt}))
                                    for arg in place:
                                        if arg != ROLE_PH_MAP['Place']:
                                            output.append((arg, 'Place', {'cor tri cnt': a_cnt}))
                            except:
                                    pass
                        used_o_cnt += 1
                    
        return output

class Life_Injure(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['injure', 'wounded', 'hurt']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
                if o_style == 'argument:sentence':
                    output_template += ' \n {}'.format('somebody or some organization led to some victim injured by some way in somewhere.')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('life event, injure sub-type')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('The event is related to life and someone is injured.')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
                if o_style == 'argument:sentence':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            " and ".join([ a['argument text'] for a in argu['Agent']]) if "Agent" in argu.keys() else ROLE_PH_MAP['Agent'],
                            " and ".join([ a['argument text'] for a in argu['Victim']]) if "Victim" in argu.keys() else ROLE_PH_MAP['Victim'],
                            " and ".join([ a['argument text'] for a in argu['Instrument']]) if "Instrument" in argu.keys() else ROLE_PH_MAP['Instrument'],
                            " and ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else ROLE_PH_MAP['Place']
                        )
                        output_texts.append("{} led to {} injured by {} in {}.".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
                
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
                    if o_style == 'argument:sentence':
                        if used_o_cnt == cnt:
                            try:
                                for a_cnt, prediction in enumerate(full_pred.split(' <sep> ')):
                                    agent = prediction.split(' led to ', 1)[0]
                                    agent = agent.split(' and ')
                                    victim = (prediction.split(' led to ', 1)[1]).split(' injured by ', 1)[0]
                                    victim = victim.split(' and ')
                                    instrument = ((prediction.split(' led to ', 1)[1]).split(' injured by ', 1)[1]).split(' in ', )[0]
                                    instrument = instrument.split(' and ')
                                    place = ((prediction.split(' led to ', 1)[1]).split(' injured by ', 1)[1]).split(' in ', 1)[1].rsplit('.',1)[0]
                                    place = place.split(' and ')
                                    for arg in agent:
                                        if arg != ROLE_PH_MAP['Agent']:
                                            output.append((arg, 'Agent', {'cor tri cnt': a_cnt}))
                                    for arg in victim:
                                        if arg != ROLE_PH_MAP['Victim']:
                                            output.append((arg, 'Victim', {'cor tri cnt': a_cnt}))
                                    for arg in instrument:
                                        if arg != ROLE_PH_MAP['Instrument']:
                                            output.append((arg, 'Instrument', {'cor tri cnt': a_cnt}))                    
                                    for arg in place:
                                        if arg != ROLE_PH_MAP['Place']:
                                            output.append((arg, 'Place', {'cor tri cnt': a_cnt}))
                            except Exception as e:
                                pass
                        used_o_cnt += 1
                    
        return output

class Life_Die(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['kill', 'death', 'assassination']
    
    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
                if o_style == 'argument:sentence':
                    output_template += ' \n {}'.format('somebody or some organization led to some victim died by some way in somewhere.')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('life event, die sub-type')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('The event is related to life and someone died.')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
                if o_style == 'argument:sentence':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            " and ".join([ a['argument text'] for a in argu['Agent']]) if "Agent" in argu.keys() else ROLE_PH_MAP['Agent'],
                            " and ".join([ a['argument text'] for a in argu['Victim']]) if "Victim" in argu.keys() else ROLE_PH_MAP['Victim'],
                            " and ".join([ a['argument text'] for a in argu['Instrument']]) if "Instrument" in argu.keys() else ROLE_PH_MAP['Instrument'],
                            " and ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else ROLE_PH_MAP['Place']
                        )
                        output_texts.append("{} led to {} died by {} in {}.".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
                
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
                    if o_style == 'argument:sentence':
                        if used_o_cnt == cnt:
                            try:
                                for a_cnt, prediction in enumerate(full_pred.split(' <sep> ')):
                                    agent = prediction.split(' led to ', 1)[0]
                                    agent = agent.split(' and ')
                                    victim = (prediction.split(' led to ', 1)[1]).split(' died by ', 1)[0]
                                    victim = victim.split(' and ')
                                    instrument = ((prediction.split(' led to ', 1)[1]).split(' died by ', 1)[1]).split(' in ', )[0]
                                    instrument = instrument.split(' and ')
                                    place = ((prediction.split(' led to ', 1)[1]).split(' died by ', 1)[1]).split(' in ', 1)[1].rsplit('.',1)[0]
                                    place = place.split(' and ')
                                    for arg in agent:
                                        if arg != ROLE_PH_MAP['Agent']:
                                            output.append((arg, 'Agent', {'cor tri cnt': a_cnt}))
                                    for arg in victim:
                                        if arg != ROLE_PH_MAP['Victim']:
                                            output.append((arg, 'Victim', {'cor tri cnt': a_cnt}))
                                    for arg in instrument:
                                        if arg != ROLE_PH_MAP['Instrument']:
                                            output.append((arg, 'Instrument', {'cor tri cnt': a_cnt}))         
                                    for arg in place:
                                        if arg != ROLE_PH_MAP['Place']:
                                            output.append((arg, 'Place', {'cor tri cnt': a_cnt}))
                            except Exception as e:
                                pass
                        used_o_cnt += 1
        return output

class Movement_Transport(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['travel', 'go', 'move']
    
    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
                if o_style == 'argument:sentence':
                    output_template += ' \n {}'.format('something was sent to somewhere from some place by some vehicle. somebody or some organization was responsible for the transport.')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('movement event, transport sub-type')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('The event is related to movement. The event occurs when a weapon or vehicle is moved from one place to another.')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
                if o_style == 'argument:sentence':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            " and ".join([ a['argument text'] for a in argu['Artifact']]) if "Artifact" in argu.keys() else ROLE_PH_MAP['Artifact'],
                            " and ".join([ a['argument text'] for a in argu['Destination']]) if "Destination" in argu.keys() else ROLE_PH_MAP['Destination'],
                            " and ".join([ a['argument text'] for a in argu['Origin']]) if "Origin" in argu.keys() else ROLE_PH_MAP['Origin'],
                            " and ".join([ a['argument text'] for a in argu['Vehicle']]) if "Vehicle" in argu.keys() else ROLE_PH_MAP['Vehicle'],
                            " and ".join([ a['argument text'] for a in argu['Agent']]) if "Agent" in argu.keys() else ROLE_PH_MAP['Agent']
                        )
                        output_texts.append("{} was sent to {} from {} by {}. {} was responsible for the transport.".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
                
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
                    if o_style == 'argument:sentence':
                        if used_o_cnt == cnt:
                            try:
                                for a_cnt, prediction in enumerate(full_pred.split(' <sep> ')):
                                    artifact = prediction.split(' was sent to ', 1)[0]
                                    artifact = artifact.split(' and ')
                                    destination = (prediction.split(' was sent to ', 1)[1]).split(' from ', 1)[0]
                                    destination = destination.split(' and ')
                                    origin = ((prediction.split(' was sent to ', 1)[1]).split(' from ', 1)[1]).split(' by ', 1)[0]
                                    origin = origin.split(' and ')
                                    vehicle = (((prediction.split(' was sent to ', 1)[1]).split(' from ', 1)[1]).split(' by ', 1)[1]).split('.', 1)[0]
                                    vehicle = vehicle.split(' and ')
                                    remain = (((prediction.split(' was sent to ', 1)[1]).split(' from ', 1)[1]).split(' by ', 1)[1]).split('.', 1)[1]
                                    if 'was responsible for the transport' in remain:
                                        agent = (remain.split(' was responsible for the transport.')[0]).strip()
                                        agent = agent.split(' and ')
                                    else:
                                        agent = []

                                    for arg in agent:
                                        if arg != ROLE_PH_MAP['Agent']:
                                            output.append((arg, 'Agent', {'cor tri cnt': a_cnt}))
                                    for arg in artifact:
                                        if arg != ROLE_PH_MAP['Artifact']:
                                            output.append((arg, 'Artifact', {'cor tri cnt': a_cnt}))
                                    for arg in destination:
                                        if arg != ROLE_PH_MAP['Destination']:
                                            output.append((arg, 'Destination', {'cor tri cnt': a_cnt}))          
                                    for arg in origin:
                                        if arg != ROLE_PH_MAP['Origin']:
                                            output.append((arg, 'Origin', {'cor tri cnt': a_cnt}))
                                    for arg in vehicle:
                                        if arg != ROLE_PH_MAP['Vehicle']:
                                            output.append((arg, 'Vehicle', {'cor tri cnt': a_cnt}))
                            except:
                                pass
                        used_o_cnt += 1
                    
        return output

class Transaction_Transfer_Ownership(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)

    @classmethod
    def get_keywords(self):
        return ['sell', 'buy', 'acquire']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
                if o_style == 'argument:sentence':
                    output_template += ' \n {}'.format('someone got something from some seller in somewhere.')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('transaction event, transfer ownership sub-type')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('The event is related to transaction. The event occurs when an item or an organization is sold or gave to some other.')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
                if o_style == 'argument:sentence':
                    output_texts = []
                    for argu in self.arguments:
                        buyer_list = []
                        if "Beneficiary" in argu.keys():
                            buyer_list.extend([ a['argument text'] for a in argu['Beneficiary']])
                        if "Buyer" in argu.keys():
                            buyer_list.extend([ a['argument text'] for a in argu['Buyer']])
                        filler = (
                            " and ".join(buyer_list) if len(buyer_list)>0 else ROLE_PH_MAP['Buyer'],
                            " and ".join([ a['argument text'] for a in argu['Artifact']]) if "Artifact" in argu.keys() else ROLE_PH_MAP['Artifact'],
                            " and ".join([ a['argument text'] for a in argu['Seller']]) if "Seller" in argu.keys() else ROLE_PH_MAP['Seller'],
                            " and ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else ROLE_PH_MAP['Place']
                        )
                        output_texts.append("{} got {} from {} in {}.".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
                
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
                    if o_style == 'argument:sentence':
                        if used_o_cnt == cnt:
                            try:
                                for a_cnt, prediction in enumerate(full_pred.split(' <sep> ')):
                                    buyer = prediction.split(' got ', 1)[0]
                                    buyer = buyer.split(' and ')
                                    remain = prediction.split(' got ', 1)[1]

                                    artifact = remain.split(' from ', 1)[0]
                                    artifact = artifact.split(' and ')
                                    remain = remain.split(' from ', 1)[1]

                                    seller = remain.split(' in ', 1)[0]
                                    seller = seller.split(' and ')
                                    remain = remain.split(' in ', 1)[1]

                                    place = remain.rsplit('.',1)[0]
                                    place = place.split(' and ')

                                    for arg in buyer:
                                        if arg != ROLE_PH_MAP['Buyer']:
                                            output.append((arg, 'Buyer', {'cor tri cnt': a_cnt}))
                                    for arg in artifact:
                                        if arg != ROLE_PH_MAP['Artifact']:
                                            output.append((arg, 'Artifact', {'cor tri cnt': a_cnt}))
                                    for arg in seller:
                                        if arg != ROLE_PH_MAP['Seller']:
                                            output.append((arg, 'Seller', {'cor tri cnt': a_cnt}))          
                                    for arg in place:
                                        if arg != ROLE_PH_MAP['Place']:
                                            output.append((arg, 'Place', {'cor tri cnt': a_cnt}))
                            except:
                                pass
                        used_o_cnt += 1
                    
        return output

class Transaction_Transfer_Money(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)

    @classmethod
    def get_keywords(self):
        return ['pay', 'donation', 'loan']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
                if o_style == 'argument:sentence':
                    output_template += ' \n {}'.format('someone paid some other in somewhere.')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('transaction event, transfer money sub-type')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('The event is related to transaction. The event occurs when someone is giving, receiving, borrowing, or lending money.')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
                if o_style == 'argument:sentence':
                    output_texts = []
                    for argu in self.arguments:
                        recipient_list = []
                        if "Recipient" in argu.keys():
                            recipient_list.extend([ a['argument text'] for a in argu['Recipient']])
                        if "Beneficiary" in argu.keys():
                            recipient_list.extend([ a['argument text'] for a in argu['Beneficiary']])
                        filler = (
                            " and ".join([ a['argument text'] for a in argu['Giver']]) if "Giver" in argu.keys() else ROLE_PH_MAP['Giver'],
                            " and ".join(recipient_list) if len(recipient_list)>0 else ROLE_PH_MAP['Recipient'],
                            " and ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else ROLE_PH_MAP['Place']
                        )
                        output_texts.append("{} paid {} in {}.".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
                
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
                    if o_style == 'argument:sentence':
                        if used_o_cnt == cnt:
                            try:
                                for a_cnt, prediction in enumerate(full_pred.split(' <sep> ')):
                                    giver = prediction.split(' paid ', 1)[0]
                                    giver = giver.split(' and ')
                                    remain = prediction.split(' paid ', 1)[1]

                                    recipient = remain.split(' in ', 1)[0]
                                    recipient = recipient.split(' and ')
                                    remain = remain.split(' in ', 1)[1]

                                    place = remain.rsplit('.',1)[0]
                                    place = place.split(' and ')

                                    for arg in giver:
                                        if arg != ROLE_PH_MAP['Giver']:
                                            output.append((arg, 'Giver', {'cor tri cnt': a_cnt}))
                                    for arg in recipient:
                                        if arg != ROLE_PH_MAP['Recipient']:
                                            output.append((arg, 'Recipient', {'cor tri cnt': a_cnt}))          
                                    for arg in place:
                                        if arg != ROLE_PH_MAP['Place']:
                                            output.append((arg, 'Place', {'cor tri cnt': a_cnt}))
                            except:
                                pass
                        used_o_cnt += 1
                    
        return output

class Business_Start_Org(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)

    @classmethod
    def get_keywords(self):
        return ['founded', 'create', 'launch']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
                if o_style == 'argument:sentence':
                    output_template += ' \n {}'.format('somebody or some organization launched some organization in somewhere.')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('business event, start organization sub-type')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('The event is related to a new organization being created.')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
                if o_style == 'argument:sentence':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            " and ".join([ a['argument text'] for a in argu['Agent']]) if "Agent" in argu.keys() else ROLE_PH_MAP['Agent'],
                            " and ".join([ a['argument text'] for a in argu['Org']]) if "Org" in argu.keys() else ROLE_PH_MAP['Org'],
                            " and ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else ROLE_PH_MAP['Place']
                        )
                        output_texts.append("{} launched {} in {}.".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
                    if o_style == 'argument:sentence':
                        if used_o_cnt == cnt:
                            try:
                                for a_cnt, prediction in enumerate(full_pred.split(' <sep> ')):
                                    agent = prediction.split(' launched ', 1)[0]
                                    agent = agent.split(' and ')
                                    remain = prediction.split(' launched ', 1)[1]

                                    org = remain.split(' in ', 1)[0]
                                    org = org.split(' and ')
                                    remain = remain.split(' in ', 1)[1]

                                    place = remain.rsplit('.',1)[0]
                                    place = place.split(' and ')
                                    for arg in agent:
                                        if arg != ROLE_PH_MAP['Agent']:
                                            output.append((arg, 'Agent', {'cor tri cnt': a_cnt}))
                                    for arg in org:
                                        if arg != ROLE_PH_MAP['Org']:
                                            output.append((arg, 'Org', {'cor tri cnt': a_cnt}))                  
                                    for arg in place:
                                        if arg != ROLE_PH_MAP['Place']:
                                            output.append((arg, 'Place', {'cor tri cnt': a_cnt}))
                            except:
                                pass
                        used_o_cnt += 1
                    
        return output

class Business_Merge_Org(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)

    @classmethod
    def get_keywords(self):
        return ['merge', 'merging', 'merger']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
                if o_style == 'argument:sentence':
                    output_template += ' \n {}'.format('some organization was merged.')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('business event, merge organization sub-type')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('The event is related to two or more organization coming together to form a new organization.')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
                if o_style == 'argument:sentence':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            " and ".join([ a['argument text'] for a in argu['Org']]) if "Org" in argu.keys() else ROLE_PH_MAP['Org']
                        )
                        output_texts.append("{} was merged.".format(filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
                
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
                    if o_style == 'argument:sentence':
                        if used_o_cnt == cnt:
                            try:
                                for a_cnt, prediction in enumerate(full_pred.split(' <sep> ')):
                                    org = prediction.split(' was merged.', 1)[0]
                                    org = org.split(' and ')
                                    for arg in org:
                                        if arg != ROLE_PH_MAP['Org']:
                                            output.append((arg, 'Org', {'cor tri cnt': a_cnt}))                  
                            except:
                                pass
                        used_o_cnt += 1
                    
        return output

class Business_Declare_Bankruptcy(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)

    @classmethod
    def get_keywords(self):
        return ['bankruptcy', 'bankrupt', 'Bankruptcy']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
                if o_style == 'argument:sentence':
                    output_template += ' \n {}'.format('some organization declared bankruptcy.')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('business event, declare bankruptcy sub-type')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('The event is related to some organization declaring bankruptcy.')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
                if o_style == 'argument:sentence':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            " and ".join([ a['argument text'] for a in argu['Org']]) if "Org" in argu.keys() else ROLE_PH_MAP['Org']
                        )
                        output_texts.append("{} declared bankruptcy.".format(filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
                
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
                    if o_style == 'argument:sentence':
                        if used_o_cnt == cnt:
                            try:
                                for a_cnt, prediction in enumerate(full_pred.split(' <sep> ')):
                                    org = prediction.split(' declared bankruptcy.', 1)[0]
                                    org = org.split(' and ')
                                    for arg in org:
                                        if arg != ROLE_PH_MAP['Org']:
                                            output.append((arg, 'Org', {'cor tri cnt': a_cnt}))                  
                            except:
                                pass
                        used_o_cnt += 1
                    
        return output

class Business_End_Org(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['dissolve', 'disbanded', 'close'] 

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
                if o_style == 'argument:sentence':
                    output_template += ' \n {}'.format('some organization dissolved.')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('business event, end organization sub-type')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('The event is related to some organization ceasing to exist.')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
                if o_style == 'argument:sentence':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            " and ".join([ a['argument text'] for a in argu['Org']]) if "Org" in argu.keys() else ROLE_PH_MAP['Org']
                        )
                        output_texts.append("{} dissolved.".format(filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
                
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
                    if o_style == 'argument:sentence':
                        if used_o_cnt == cnt:
                            try:
                                for a_cnt, prediction in enumerate(full_pred.split(' <sep> ')):
                                    org = prediction.split(' dissolved.', 1)[0]
                                    org = org.split(' and ')
                                    for arg in org:
                                        if arg != ROLE_PH_MAP['Org']:
                                            output.append((arg, 'Org', {'cor tri cnt': a_cnt}))                  
                            except:
                                pass
                        used_o_cnt += 1
                    
        return output

class Conflict_Attack(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)

    @classmethod
    def get_keywords(self):
        return ['war', 'attack', 'terrorism']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
                if o_style == 'argument:sentence':
                    output_template += ' \n {}'.format('some attacker attacked some facility, someone, or some organization by some way in somewhere.')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('conflict event, attack sub-type')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('The event is related to conflict and some violent physical act.')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
                if o_style == 'argument:sentence':
                    output_texts = []
                    for argu in self.arguments:
                        attacker_list = []
                        if "Attacker" in argu.keys():
                            attacker_list.extend([ a['argument text'] for a in argu['Attacker']])
                        if "Agent" in argu.keys():
                            attacker_list.extend([ a['argument text'] for a in argu['Agent']])  
                        filler = (
                            " and ".join(attacker_list) if len(attacker_list)>0 else ROLE_PH_MAP['Attacker'],
                            " and ".join([ a['argument text'] for a in argu['Target']]) if "Target" in argu.keys() else ROLE_PH_MAP['Target'],
                            " and ".join([ a['argument text'] for a in argu['Instrument']]) if "Instrument" in argu.keys() else ROLE_PH_MAP['Instrument'],
                            " and ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else ROLE_PH_MAP['Place']
                        )
                        output_texts.append("{} attacked {} by {} in {}.".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
                
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except Exception as e:
                                #print(e)
                                pass
                        used_o_cnt += 1
                    if o_style == 'argument:sentence':
                        if used_o_cnt == cnt:
                            try:
                                for a_cnt, prediction in enumerate(full_pred.split(' <sep> ')):
                                    attack = prediction.split(' attacked ', 1)[0]
                                    attack = attack.split(' and ')
                                    remain = prediction.split(' attacked ', 1)[1]

                                    target = remain.split(' by ', 1)[0]
                                    target = target.split(' and ')
                                    remain = remain.split(' by ', 1)[1]

                                    instrument = remain.split(' in ', 1)[0]
                                    instrument = instrument.split(' and ')
                                    remain = remain.split(' in ', 1)[1]

                                    place = remain.rsplit('.',1)[0]
                                    place = place.split(' and ')
                                    for arg in attack:
                                        if arg != ROLE_PH_MAP['Attacker']:
                                            output.append((arg, 'Attacker', {'cor tri cnt': a_cnt}))
                                    for arg in target:
                                        if arg != ROLE_PH_MAP['Target']:
                                            output.append((arg, 'Target', {'cor tri cnt': a_cnt}))
                                    for arg in instrument:
                                        if arg != ROLE_PH_MAP['Instrument']:
                                            output.append((arg, 'Instrument', {'cor tri cnt': a_cnt}))                    
                                    for arg in place:
                                        if arg != ROLE_PH_MAP['Place']:
                                            output.append((arg, 'Place', {'cor tri cnt': a_cnt}))
                            except:
                                pass
                        used_o_cnt += 1
                    
        return output

class Conflict_Demonstrate(event_template):
    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)

    @classmethod
    def get_keywords(self):
        return ['rally', 'protest', 'demonstrate']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
                if o_style == 'argument:sentence':
                    output_template += ' \n {}'.format('some people or some organization protest at somewhere.')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('conflict event, demonstrate sub-type')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('The event is related to a large number of people coming together to protest.')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
                if o_style == 'argument:sentence':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            " and ".join([ a['argument text'] for a in argu['Entity']]) if "Entity" in argu.keys() else ROLE_PH_MAP['Entity'],
                            " and ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else ROLE_PH_MAP['Place']
                        )
                        output_texts.append("{} protest at {}.".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
                    if o_style == 'argument:sentence':
                        if used_o_cnt == cnt:
                            try:
                                for a_cnt, prediction in enumerate(full_pred.split(' <sep> ')):
                                    entity = prediction.split(' protest at ', 1)[0]
                                    entity = entity.split(' and ')
                                    remain = prediction.split(' protest at ', 1)[1]

                                    place = remain.rsplit('.',1)[0]
                                    place = place.split(' and ')
                                    for arg in entity:
                                        if arg != ROLE_PH_MAP['Entity']:
                                            output.append((arg, 'Entity', {'cor tri cnt': a_cnt}))
                                    for arg in place:
                                        if arg != ROLE_PH_MAP['Place']:
                                            output.append((arg, 'Place', {'cor tri cnt': a_cnt}))
                            except:
                                pass                                    
                        used_o_cnt += 1
                    
        return output

class Contact_Meet(event_template):
    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)

    @classmethod
    def get_keywords(self):
        return ['meeting', 'met', 'summit']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
                if o_style == 'argument:sentence':
                    output_template += ' \n {}'.format('some people or some organization met at somewhere.')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('contact event, meet sub-type')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('The event is related to a group of people meeting and interacting with one another face-to-face.')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
                if o_style == 'argument:sentence':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            " and ".join([ a['argument text'] for a in argu['Entity']]) if "Entity" in argu.keys() else ROLE_PH_MAP['Entity'],
                            " and ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else ROLE_PH_MAP['Place']
                        )
                        output_texts.append("{} met at {}.".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
                
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
                    if o_style == 'argument:sentence':
                        if used_o_cnt == cnt:
                            try:
                                for a_cnt, prediction in enumerate(full_pred.split(' <sep> ')):
                                    entity = prediction.split(' met at ', 1)[0]
                                    entity = entity.split(' and ')
                                    remain = prediction.split(' met at ', 1)[1]

                                    place = remain.rsplit('.',1)[0]
                                    place = place.split(' and ')
                                    for arg in entity:
                                        if arg != ROLE_PH_MAP['Entity']:
                                            output.append((arg, 'Entity', {'cor tri cnt': a_cnt}))
                                    for arg in place:
                                        if arg != ROLE_PH_MAP['Place']:
                                            output.append((arg, 'Place', {'cor tri cnt': a_cnt}))
                            except:
                                pass                                    
                        used_o_cnt += 1
                    
        return output
    
class Contact_Phone_Write(event_template):
    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)

    @classmethod
    def get_keywords(self):
        return ['call', 'communicate', 'e-mail']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
                if o_style == 'argument:sentence':
                    output_template += ' \n {}'.format('some people or some organization called or texted messages at somewhere.')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('contact event, phone write sub-type')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('The event is related to people phone calling or messaging one another.')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
                if o_style == 'argument:sentence':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            " and ".join([ a['argument text'] for a in argu['Entity']]) if "Entity" in argu.keys() else ROLE_PH_MAP['Entity'],
                            " and ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else ROLE_PH_MAP['Place']
                        )
                        output_texts.append("{} called or texted messages at {}.".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
                
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
                    if o_style == 'argument:sentence':
                        if used_o_cnt == cnt:
                            try:
                                for a_cnt, prediction in enumerate(full_pred.split(' <sep> ')):
                                    entity = prediction.split(' called or texted messages at ', 1)[0]
                                    entity = entity.split(' and ')
                                    remain = prediction.split(' called or texted messages at ', 1)[1]

                                    place = remain.rsplit('.',1)[0]
                                    place = place.split(' and ')
                                    for arg in entity:
                                        if arg != ROLE_PH_MAP['Entity']:
                                            output.append((arg, 'Entity', {'cor tri cnt': a_cnt}))
                                    for arg in place:
                                        if arg != ROLE_PH_MAP['Place']:
                                            output.append((arg, 'Place', {'cor tri cnt': a_cnt}))
                            except:
                                pass                                   
                        used_o_cnt += 1
                    
        return output
            
class Personnel_Start_Position(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)

    @classmethod
    def get_keywords(self):
        return ['hire', 'appoint', 'join']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
                if o_style == 'argument:sentence':
                    output_template += ' \n {}'.format('somebody got new job and was hired by some people or some organization in somewhere.')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('personnel event, start position sub-type')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('The event is related to a person begins working for an organization or a hiring manager.')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
                if o_style == 'argument:sentence':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            " and ".join([ a['argument text'] for a in argu['Person']]) if "Person" in argu.keys() else ROLE_PH_MAP['Person'],
                            " and ".join([ a['argument text'] for a in argu['Entity']]) if "Entity" in argu.keys() else ROLE_PH_MAP['Entity'],
                            " and ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else ROLE_PH_MAP['Place']
                        )
                        output_texts.append("{} got new job and was hired by {} in {}.".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
                    if o_style == 'argument:sentence':
                        if used_o_cnt == cnt:
                            try:
                                for a_cnt, prediction in enumerate(full_pred.split(' <sep> ')):
                                    person = prediction.split(' got new job and was hired by ', 1)[0]
                                    person = person.split(' and ')
                                    remain = prediction.split(' got new job and was hired by ', 1)[1]

                                    entity = remain.split(' in ', 1)[0]
                                    entity = entity.split(' and ')
                                    remain = remain.split(' in ', 1)[1]

                                    place = remain.rsplit('.',1)[0]
                                    place = place.split(' and ')
                                    for arg in person:
                                        if arg != ROLE_PH_MAP['Person']:
                                            output.append((arg, 'Person', {'cor tri cnt': a_cnt}))
                                    for arg in entity:
                                        if arg != ROLE_PH_MAP['Entity']:
                                            output.append((arg, 'Entity', {'cor tri cnt': a_cnt}))
                                    for arg in place:
                                        if arg != ROLE_PH_MAP['Place']:
                                            output.append((arg, 'Place', {'cor tri cnt': a_cnt}))
                            except:
                                pass
                        used_o_cnt += 1
        return output

class Personnel_End_Position(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)

    @classmethod
    def get_keywords(self):
        return ['former', 'laid off', 'fired']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
                if o_style == 'argument:sentence':
                    output_template += ' \n {}'.format('somebody stopped working for some people or some organization at somewhere.')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('personnel event, end position sub-type')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('The event is related to a person stops working for an organization or a hiring manager.')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
                if o_style == 'argument:sentence':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            " and ".join([ a['argument text'] for a in argu['Person']]) if "Person" in argu.keys() else ROLE_PH_MAP['Person'],
                            " and ".join([ a['argument text'] for a in argu['Entity']]) if "Entity" in argu.keys() else ROLE_PH_MAP['Entity'],
                            " and ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else ROLE_PH_MAP['Place']
                        )
                        output_texts.append("{} stopped working for {} at {}.".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
                    if o_style == 'argument:sentence':
                        if used_o_cnt == cnt:
                            try:
                                for a_cnt, prediction in enumerate(full_pred.split(' <sep> ')):
                                    person = prediction.split(' stopped working for ', 1)[0]
                                    person = person.split(' and ')
                                    remain = prediction.split(' stopped working for ', 1)[1]

                                    entity = remain.split(' at ', 1)[0]
                                    entity = entity.split(' and ')
                                    remain = remain.split(' at ', 1)[1]

                                    place = remain.rsplit('.',1)[0]
                                    place = place.split(' and ')
                                    for arg in person:
                                        if arg != ROLE_PH_MAP['Person']:
                                            output.append((arg, 'Person', {'cor tri cnt': a_cnt}))
                                    for arg in entity:
                                        if arg != ROLE_PH_MAP['Entity']:
                                            output.append((arg, 'Entity', {'cor tri cnt': a_cnt}))
                                    for arg in place:
                                        if arg != ROLE_PH_MAP['Place']:
                                            output.append((arg, 'Place', {'cor tri cnt': a_cnt}))
                            except:
                                pass
                        used_o_cnt += 1
        return output

class Personnel_Nominate(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)

    @classmethod
    def get_keywords(self):
        return ['named', 'nomination', 'nominate']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
                if o_style == 'argument:sentence':
                    output_template += ' \n {}'.format('somebody was nominated by somebody or some organization to do a job.')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('personnel event, nominate position sub-type')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('The event is related to a person being nominated for a position.')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
                if o_style == 'argument:sentence':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            " and ".join([ a['argument text'] for a in argu['Person']]) if "Person" in argu.keys() else ROLE_PH_MAP['Person'],
                            " and ".join([ a['argument text'] for a in argu['Agent']]) if "Agent" in argu.keys() else ROLE_PH_MAP['Agent']
                        )
                        output_texts.append("{} was nominated by {} to do a job.".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
                    if o_style == 'argument:sentence':
                        if used_o_cnt == cnt:
                            try:
                                for a_cnt, prediction in enumerate(full_pred.split(' <sep> ')):
                                    person = prediction.split(' was nominated by ', 1)[0]
                                    person = person.split(' and ')
                                    remain = prediction.split(' was nominated by ', 1)[1]

                                    agent = remain.rsplit(' to do a job.',1)[0]
                                    agent = agent.split(' and ')
                                    for arg in person:
                                        if arg != ROLE_PH_MAP['Person']:
                                            output.append((arg, 'Person', {'cor tri cnt': a_cnt}))
                                    for arg in agent:
                                        if arg != ROLE_PH_MAP['Agent']:
                                            output.append((arg, 'Agent', {'cor tri cnt': a_cnt}))
                            except:
                                pass
                        used_o_cnt += 1
        return output

class Personnel_Elect(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['election', 'elect', 'elected']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
                if o_style == 'argument:sentence':
                    output_template += ' \n {}'.format('somebody was elected a position, and the election was voted by some people or some organization in somewhere.')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('personnel event, elect position sub-type')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('The event is related to a candidate wins an election.')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
                if o_style == 'argument:sentence':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            " and ".join([ a['argument text'] for a in argu['Person']]) if "Person" in argu.keys() else ROLE_PH_MAP['Person'],
                            " and ".join([ a['argument text'] for a in argu['Entity']]) if "Entity" in argu.keys() else ROLE_PH_MAP['Entity'],
                            " and ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else ROLE_PH_MAP['Place']
                        )
                        output_texts.append("{} was elected a position, and the election was voted by {} in {}.".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
                    if o_style == 'argument:sentence':
                        if used_o_cnt == cnt:
                            try:
                                for a_cnt, prediction in enumerate(full_pred.split(' <sep> ')):
                                    person = prediction.split(' was elected a position, and the election was voted by ', 1)[0]
                                    person = person.split(' and ')
                                    remain = prediction.split(' was elected a position, and the election was voted by ', 1)[1]

                                    entity = remain.split(' in ', 1)[0]
                                    entity = entity.split(' and ')
                                    remain = remain.split(' in ', 1)[1]

                                    place = remain.rsplit('.',1)[0]
                                    place = place.split(' and ')
                                    for arg in person:
                                        if arg != ROLE_PH_MAP['Person']:
                                            output.append((arg, 'Person', {'cor tri cnt': a_cnt}))
                                    for arg in entity:
                                        if arg != ROLE_PH_MAP['Entity']:
                                            output.append((arg, 'Entity', {'cor tri cnt': a_cnt}))
                                    for arg in place:
                                        if arg != ROLE_PH_MAP['Place']:
                                            output.append((arg, 'Place', {'cor tri cnt': a_cnt}))
                            except:
                                pass
                        used_o_cnt += 1
        return output

class Justice_Arrest_Jail(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)

    @classmethod
    def get_keywords(self):
        return ['arrest', 'jail', 'detained']
    
    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
                if o_style == 'argument:sentence':
                    output_template += ' \n {}'.format('somebody was sent to jailed or arrested by somebody or some organization in somewhere.')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('justice event, arrest jail sub-type')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('The event is related to a person getting arrested or a person being sent to jail.')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
                if o_style == 'argument:sentence':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            " and ".join([ a['argument text'] for a in argu['Person']]) if "Person" in argu.keys() else ROLE_PH_MAP['Person'],
                            " and ".join([ a['argument text'] for a in argu['Agent']]) if "Agent" in argu.keys() else ROLE_PH_MAP['Agent'],
                            " and ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else ROLE_PH_MAP['Place']
                        )
                        output_texts.append("{} was sent to jailed or arrested by {} in {}.".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
                    if o_style == 'argument:sentence':
                        if used_o_cnt == cnt:
                            try:
                                for a_cnt, prediction in enumerate(full_pred.split(' <sep> ')):
                                    person = prediction.split(' was sent to jailed or arrested by ', 1)[0]
                                    person = person.split(' and ')
                                    remain = prediction.split(' was sent to jailed or arrested by ', 1)[1]

                                    agent = remain.split(' in ', 1)[0]
                                    agent = agent.split(' and ')
                                    remain = remain.split(' in ', 1)[1]

                                    place = remain.rsplit('.',1)[0]
                                    place = place.split(' and ')
                                    for arg in person:
                                        if arg != ROLE_PH_MAP['Person']:
                                            output.append((arg, 'Person', {'cor tri cnt': a_cnt}))
                                    for arg in agent:
                                        if arg != ROLE_PH_MAP['Agent']:
                                            output.append((arg, 'Agent', {'cor tri cnt': a_cnt}))
                                    for arg in place:
                                        if arg != ROLE_PH_MAP['Place']:
                                            output.append((arg, 'Place', {'cor tri cnt': a_cnt}))
                            except Exception as e:
                                pass
                        used_o_cnt += 1
        return output

class Justice_Release_Parole(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)

    @classmethod
    def get_keywords(self):
        return ['parole', 'release', 'free']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
                if o_style == 'argument:sentence':
                    output_template += ' \n {}'.format('somebody was released by some people or some organization from somewhere.')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('justice event, release parole sub-type')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format("The event is related to an end to someone's custody in prison.")
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
                if o_style == 'argument:sentence':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            " and ".join([ a['argument text'] for a in argu['Person']]) if "Person" in argu.keys() else ROLE_PH_MAP['Person'],
                            " and ".join([ a['argument text'] for a in argu['Entity']]) if "Entity" in argu.keys() else ROLE_PH_MAP['Entity'],
                            " and ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else ROLE_PH_MAP['Place']
                        )
                        output_texts.append("{} was released by {} from {}.".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
                    if o_style == 'argument:sentence':
                        if used_o_cnt == cnt:
                            try:
                                for a_cnt, prediction in enumerate(full_pred.split(' <sep> ')):
                                    person = prediction.split(' was released by ', 1)[0]
                                    person = person.split(' and ')
                                    remain = prediction.split(' was released by ', 1)[1]

                                    entity = remain.split(' from ', 1)[0]
                                    entity = entity.split(' and ')
                                    remain = remain.split(' from ', 1)[1]

                                    place = remain.rsplit('.',1)[0]
                                    place = place.split(' and ')
                                    for arg in person:
                                        if arg != ROLE_PH_MAP['Person']:
                                            output.append((arg, 'Person', {'cor tri cnt': a_cnt}))
                                    for arg in entity:
                                        if arg != ROLE_PH_MAP['Entity']:
                                            output.append((arg, 'Entity', {'cor tri cnt': a_cnt}))
                                    for arg in place:
                                        if arg != ROLE_PH_MAP['Place']:
                                            output.append((arg, 'Place', {'cor tri cnt': a_cnt}))
                            except:
                                pass
                        used_o_cnt += 1
        return output

class Justice_Trial_Hearing(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['trial', 'hearing', 'proceeding']    
    
    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
                if o_style == 'argument:sentence':
                    output_template += ' \n {}'.format('somebody, prosecuted by some other, faced a trial in somewhere. The hearing was judged by some adjudicator.')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('justice event, trial hearing sub-type')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format("The event is related to a trial or hearing for someone.")
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
                if o_style == 'argument:sentence':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            " and ".join([ a['argument text'] for a in argu['Defendant']]) if "Defendant" in argu.keys() else ROLE_PH_MAP['Defendant'],
                            " and ".join([ a['argument text'] for a in argu['Prosecutor']]) if "Prosecutor" in argu.keys() else ROLE_PH_MAP['Prosecutor'],
                            " and ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else ROLE_PH_MAP['Place'],
                            " and ".join([ a['argument text'] for a in argu['Adjudicator']]) if "Adjudicator" in argu.keys() else ROLE_PH_MAP['Adjudicator']
                        )
                        output_texts.append("{}, prosecuted by {}, faced a trial in {}. The hearing was judged by {}.".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
                    if o_style == 'argument:sentence':
                        if used_o_cnt == cnt:
                            try:
                                for a_cnt, prediction in enumerate(full_pred.split(' <sep> ')):
                                    defendant = prediction.split(', prosecuted by ', 1)[0]
                                    defendant = defendant.split(' and ')
                                    remain = prediction.split(', prosecuted by ', 1)[1]

                                    prosecutor = remain.split(', faced a trial in ', 1)[0]
                                    prosecutor = prosecutor.split(' and ')
                                    remain = remain.split(', faced a trial in ', 1)[1]

                                    place = remain.split('. The hearing was judged by ', 1)[0]
                                    place = place.split(' and ')
                                    remain = remain.split('. The hearing was judged by ', 1)[1]

                                    adjudicator = remain.rsplit('.',1)[0]
                                    adjudicator = adjudicator.split(' and ')
                                    for arg in defendant:
                                        if arg != ROLE_PH_MAP['Defendant']:
                                            output.append((arg, 'Defendant', {'cor tri cnt': a_cnt}))
                                    for arg in prosecutor:
                                        if arg != ROLE_PH_MAP['Prosecutor']:
                                            output.append((arg, 'Prosecutor', {'cor tri cnt': a_cnt}))
                                    for arg in place:
                                        if arg != ROLE_PH_MAP['Place']:
                                            output.append((arg, 'Place', {'cor tri cnt': a_cnt}))
                                    for arg in adjudicator:
                                        if arg != ROLE_PH_MAP['Adjudicator']:
                                            output.append((arg, 'Adjudicator', {'cor tri cnt': a_cnt}))
                            except Exception as e:
                                pass
                        used_o_cnt += 1
        return output

class Justice_Charge_Indict(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)

    @classmethod
    def get_keywords(self):
        return ['indict', 'charged', 'accused']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
                if o_style == 'argument:sentence':
                    output_template += ' \n {}'.format('somebody was charged by some other in somewhere. The adjudication was judged by some adjudicator.')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('justice event, charge indict sub-type')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format("The event is related to someone or some organization being accused of a crime.")
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
                if o_style == 'argument:sentence':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            " and ".join([ a['argument text'] for a in argu['Defendant']]) if "Defendant" in argu.keys() else ROLE_PH_MAP['Defendant'],
                            " and ".join([ a['argument text'] for a in argu['Prosecutor']]) if "Prosecutor" in argu.keys() else ROLE_PH_MAP['Prosecutor'],
                            " and ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else ROLE_PH_MAP['Place'],
                            " and ".join([ a['argument text'] for a in argu['Adjudicator']]) if "Adjudicator" in argu.keys() else ROLE_PH_MAP['Adjudicator']
                        )
                        output_texts.append("{} was charged by {} in {}. The adjudication was judged by {}.".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
                    if o_style == 'argument:sentence':
                        if used_o_cnt == cnt:
                            try:
                                for a_cnt, prediction in enumerate(full_pred.split(' <sep> ')):
                                    defendant = prediction.split(' was charged by ', 1)[0]
                                    defendant = defendant.split(' and ')
                                    remain = prediction.split(' was charged by ', 1)[1]

                                    prosecutor = remain.split(' in ', 1)[0]
                                    prosecutor = prosecutor.split(' and ')
                                    remain = remain.split(' in ', 1)[1]

                                    place = remain.split('. The adjudication was judged by ', 1)[0]
                                    place = place.split(' and ')
                                    remain = remain.split('. The adjudication was judged by ', 1)[1]

                                    adjudicator = remain.rsplit('.',1)[0]
                                    adjudicator = adjudicator.split(' and ')
                                    for arg in defendant:
                                        if arg != ROLE_PH_MAP['Defendant']:
                                            output.append((arg, 'Defendant', {'cor tri cnt': a_cnt}))
                                    for arg in prosecutor:
                                        if arg != ROLE_PH_MAP['Prosecutor']:
                                            output.append((arg, 'Prosecutor', {'cor tri cnt': a_cnt}))
                                    for arg in place:
                                        if arg != ROLE_PH_MAP['Place']:
                                            output.append((arg, 'Place', {'cor tri cnt': a_cnt}))
                                    for arg in adjudicator:
                                        if arg != ROLE_PH_MAP['Adjudicator']:
                                            output.append((arg, 'Adjudicator', {'cor tri cnt': a_cnt}))
                            except:
                                pass
                        used_o_cnt += 1
        return output

class Justice_Sue(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)

    @classmethod
    def get_keywords(self):
        return ['sue', 'lawsuit', 'suit']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
                if o_style == 'argument:sentence':
                    output_template += ' \n {}'.format('somebody was sued by some other in somewhere. The adjudication was judged by some adjudicator.')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('justice event, sue sub-type')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format("The event is related to a court proceeding that has been initiated and someone sue the other.")
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
                if o_style == 'argument:sentence':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            " and ".join([ a['argument text'] for a in argu['Defendant']]) if "Defendant" in argu.keys() else ROLE_PH_MAP['Defendant'],
                            " and ".join([ a['argument text'] for a in argu['Plaintiff']]) if "Plaintiff" in argu.keys() else ROLE_PH_MAP['Plaintiff'],
                            " and ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else ROLE_PH_MAP['Place'],
                            " and ".join([ a['argument text'] for a in argu['Adjudicator']]) if "Adjudicator" in argu.keys() else ROLE_PH_MAP['Adjudicator']
                        )
                        output_texts.append("{} was sued by {} in {}. The adjudication was judged by {}.".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
                    if o_style == 'argument:sentence':
                        if used_o_cnt == cnt:
                            try:
                                for a_cnt, prediction in enumerate(full_pred.split(' <sep> ')):
                                    defendant = prediction.split(' was sued by ', 1)[0]
                                    defendant = defendant.split(' and ')
                                    remain = prediction.split(' was sued by ', 1)[1]

                                    plaintiff = remain.split(' in ', 1)[0]
                                    plaintiff = plaintiff.split(' and ')
                                    remain = remain.split(' in ', 1)[1]

                                    place = remain.split('. The adjudication was judged by ', 1)[0]
                                    place = place.split(' and ')
                                    remain = remain.split('. The adjudication was judged by ', 1)[1]

                                    adjudicator = remain.rsplit('.',1)[0]
                                    adjudicator = adjudicator.split(' and ')
                                    for arg in defendant:
                                        if arg != ROLE_PH_MAP['Defendant']:
                                            output.append((arg, 'Defendant', {'cor tri cnt': a_cnt}))
                                    for arg in plaintiff:
                                        if arg != ROLE_PH_MAP['Plaintiff']:
                                            output.append((arg, 'Plaintiff', {'cor tri cnt': a_cnt}))
                                    for arg in place:
                                        if arg != ROLE_PH_MAP['Place']:
                                            output.append((arg, 'Place', {'cor tri cnt': a_cnt}))
                                    for arg in adjudicator:
                                        if arg != ROLE_PH_MAP['Adjudicator']:
                                            output.append((arg, 'Adjudicator', {'cor tri cnt': a_cnt}))
                            except:
                                pass
                        used_o_cnt += 1
        return output

class Justice_Convict(event_template):

    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)

    @classmethod
    def get_keywords(self):
        return ['convicted', 'guilty', 'verdict']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
                if o_style == 'argument:sentence':
                    output_template += ' \n {}'.format('somebody was convicted of a crime in somewhere. The adjudication was judged by some adjudicator.')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('justice event, convict sub-type')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format("The event is related to someone being found guilty of a crime.")
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
                if o_style == 'argument:sentence':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            " and ".join([ a['argument text'] for a in argu['Defendant']]) if "Defendant" in argu.keys() else ROLE_PH_MAP['Defendant'],
                            " and ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else ROLE_PH_MAP['Place'],
                            " and ".join([ a['argument text'] for a in argu['Adjudicator']]) if "Adjudicator" in argu.keys() else ROLE_PH_MAP['Adjudicator']
                        )
                        output_texts.append("{} was convicted of a crime in {}. The adjudication was judged by {}.".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
                    if o_style == 'argument:sentence':
                        if used_o_cnt == cnt:
                            try:
                                for a_cnt, prediction in enumerate(full_pred.split(' <sep> ')):
                                    defendant = prediction.split(' was convicted of a crime in ', 1)[0]
                                    defendant = defendant.split(' and ')
                                    remain = prediction.split(' was convicted of a crime in ', 1)[1]

                                    place = remain.split('. The adjudication was judged by ', 1)[0]
                                    place = place.split(' and ')
                                    remain = remain.split('. The adjudication was judged by ', 1)[1]

                                    adjudicator = remain.rsplit('.',1)[0]
                                    adjudicator = adjudicator.split(' and ')
                                    for arg in defendant:
                                        if arg != ROLE_PH_MAP['Defendant']:
                                            output.append((arg, 'Defendant', {'cor tri cnt': a_cnt}))
                                    for arg in place:
                                        if arg != ROLE_PH_MAP['Place']:
                                            output.append((arg, 'Place', {'cor tri cnt': a_cnt}))
                                    for arg in adjudicator:
                                        if arg != ROLE_PH_MAP['Adjudicator']:
                                            output.append((arg, 'Adjudicator', {'cor tri cnt': a_cnt}))
                            except:
                                pass
                        used_o_cnt += 1
        return output

class Justice_Sentence(event_template):
    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)

    @classmethod
    def get_keywords(self):
        return ['sentenced', 'sentencing', 'sentence']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
                if o_style == 'argument:sentence':
                    output_template += ' \n {}'.format('somebody was sentenced to punishment in somewhere. The adjudication was judged by some adjudicator.')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('justice event, sentence sub-type')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format("The event is related to someone being sentenced to punishment because of a crime.")
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
                if o_style == 'argument:sentence':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            " and ".join([ a['argument text'] for a in argu['Defendant']]) if "Defendant" in argu.keys() else ROLE_PH_MAP['Defendant'],
                            " and ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else ROLE_PH_MAP['Place'],
                            " and ".join([ a['argument text'] for a in argu['Adjudicator']]) if "Adjudicator" in argu.keys() else ROLE_PH_MAP['Adjudicator']
                        )
                        output_texts.append("{} was sentenced to punishment in {}. The adjudication was judged by {}.".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
                    if o_style == 'argument:sentence':
                        if used_o_cnt == cnt:
                            try:
                                for a_cnt, prediction in enumerate(full_pred.split(' <sep> ')):
                                    defendant = prediction.split(' was sentenced to punishment in ', 1)[0]
                                    defendant = defendant.split(' and ')
                                    remain = prediction.split(' was sentenced to punishment in ', 1)[1]

                                    place = remain.split('. The adjudication was judged by ', 1)[0]
                                    place = place.split(' and ')
                                    remain = remain.split('. The adjudication was judged by ', 1)[1]

                                    adjudicator = remain.rsplit('.',1)[0]
                                    adjudicator = adjudicator.split(' and ')
                                    for arg in defendant:
                                        if arg != ROLE_PH_MAP['Defendant']:
                                            output.append((arg, 'Defendant', {'cor tri cnt': a_cnt}))
                                    for arg in place:
                                        if arg != ROLE_PH_MAP['Place']:
                                            output.append((arg, 'Place', {'cor tri cnt': a_cnt}))
                                    for arg in adjudicator:
                                        if arg != ROLE_PH_MAP['Adjudicator']:
                                            output.append((arg, 'Adjudicator', {'cor tri cnt': a_cnt}))
                            except:
                                pass
                        used_o_cnt += 1
        return output

class Justice_Fine(event_template):
    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)

    @classmethod
    def get_keywords(self):
        return ['fine', 'fined', 'payouts']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
                if o_style == 'argument:sentence':
                    output_template += ' \n {}'.format('some people or some organization in somewhere was ordered by some adjudicator to pay a fine.')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('justice event, fine sub-type')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format("The event is related to someone being issued a financial punishment.")
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
                if o_style == 'argument:sentence':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            " and ".join([ a['argument text'] for a in argu['Entity']]) if "Entity" in argu.keys() else ROLE_PH_MAP['Entity'],
                            " and ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else ROLE_PH_MAP['Place'],
                            " and ".join([ a['argument text'] for a in argu['Adjudicator']]) if "Adjudicator" in argu.keys() else ROLE_PH_MAP['Adjudicator']
                        )
                        output_texts.append("{} in {} was ordered by {} to pay a fine.".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
                    if o_style == 'argument:sentence':
                        if used_o_cnt == cnt:
                            try:
                                for a_cnt, prediction in enumerate(full_pred.split(' <sep> ')):
                                    entity = prediction.split(' in ', 1)[0]
                                    entity = entity.split(' and ')
                                    remain = prediction.split(' in ', 1)[1]

                                    place = remain.split(' was ordered by ', 1)[0]
                                    place = place.split(' and ')
                                    remain = remain.split(' was ordered by ', 1)[1]

                                    adjudicator = remain.rsplit(' to pay a fine.',1)[0]
                                    adjudicator = adjudicator.split(' and ')
                                    for arg in entity:
                                        if arg != ROLE_PH_MAP['Entity']:
                                            output.append((arg, 'Entity', {'cor tri cnt': a_cnt}))
                                    for arg in place:
                                        if arg != ROLE_PH_MAP['Place']:
                                            output.append((arg, 'Place', {'cor tri cnt': a_cnt}))
                                    for arg in adjudicator:
                                        if arg != ROLE_PH_MAP['Adjudicator']:
                                            output.append((arg, 'Adjudicator', {'cor tri cnt': a_cnt}))
                            except Exception as e:
                                pass
                        used_o_cnt += 1
        return output

class Justice_Execute(event_template):
    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)

    @classmethod
    def get_keywords(self):
        return ['execution', 'executed', 'execute']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
                if o_style == 'argument:sentence':
                    output_template += ' \n {}'.format('somebody was executed by somebody or some organization at somewhere.')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('justice event, execute sub-type')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format("The event is related to someone being executed to death.")
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
                if o_style == 'argument:sentence':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            " and ".join([ a['argument text'] for a in argu['Person']]) if "Person" in argu.keys() else ROLE_PH_MAP['Person'],
                            " and ".join([ a['argument text'] for a in argu['Agent']]) if "Agent" in argu.keys() else ROLE_PH_MAP['Agent'],
                            " and ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else ROLE_PH_MAP['Place']
                        )
                        output_texts.append("{} was executed by {} at {}.".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
                    if o_style == 'argument:sentence':
                        if used_o_cnt == cnt:
                            try:
                                for a_cnt, prediction in enumerate(full_pred.split(' <sep> ')):
                                    person = prediction.split(' was executed by ', 1)[0]
                                    person = person.split(' and ')
                                    remain = prediction.split(' was executed by ', 1)[1]

                                    agent = remain.split(' at ', 1)[0]
                                    agent = agent.split(' and ')
                                    remain = remain.split(' at ', 1)[1]

                                    place = remain.rsplit('.',1)[0]
                                    place = place.split(' and ')
                                    for arg in person:
                                        if arg != ROLE_PH_MAP['Person']:
                                            output.append((arg, 'Person', {'cor tri cnt': a_cnt}))
                                    for arg in place:
                                        if arg != ROLE_PH_MAP['Place']:
                                            output.append((arg, 'Place', {'cor tri cnt': a_cnt}))
                                    for arg in agent:
                                        if arg != ROLE_PH_MAP['Agent']:
                                            output.append((arg, 'Agent', {'cor tri cnt': a_cnt}))
                            except:
                                pass
                        used_o_cnt += 1
        return output
            
class Justice_Extradite(event_template):
    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)

    @classmethod
    def get_keywords(self):
        return ['extradition', 'extradited', 'extraditing']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
                if o_style == 'argument:sentence':
                    output_template += ' \n {}'.format('somebody was extradicted to somewhere from some place. somebody or some organization was responsible for the extradition.')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('justice event, extradite sub-type')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format('The event is related to justice. The event occurs when a person was extradited from one place to another place.')
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
                if o_style == 'argument:sentence':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            " and ".join([ a['argument text'] for a in argu['Person']]) if "Person" in argu.keys() else ROLE_PH_MAP['Person'],
                            " and ".join([ a['argument text'] for a in argu['Destination']]) if "Destination" in argu.keys() else ROLE_PH_MAP['Destination'],
                            " and ".join([ a['argument text'] for a in argu['Origin']]) if "Origin" in argu.keys() else ROLE_PH_MAP['Origin'],
                            " and ".join([ a['argument text'] for a in argu['Agent']]) if "Agent" in argu.keys() else ROLE_PH_MAP['Agent']
                        )
                        output_texts.append("{} was extradicted to {} from {}. {} was responsible for the extradition.".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
                    if o_style == 'argument:sentence':
                        if used_o_cnt == cnt:
                            try:
                                for a_cnt, prediction in enumerate(full_pred.split(' <sep> ')):
                                    person = prediction.split(' was extradicted to ')[0]
                                    person = person.split(' and ')

                                    destination = (prediction.split(' was extradicted to ')[1]).split(' from ')[0]
                                    destination = destination.split(' and ')

                                    origin = ((prediction.split(' was extradicted to ')[1]).split(' from ')[1]).split('.', 1)[0]
                                    origin = origin.split(' and ')

                                    remain = ((prediction.split(' was extradicted to ')[1]).split(' from ')[1]).split('.', 1)[1]

                                    if 'was responsible for' in remain:
                                        agent = (remain.split(' was responsible for the extradition.')[0]).strip()
                                        agent = agent.split(' and ')
                                    else:
                                        agent = []

                                    for arg in agent:
                                        if arg != ROLE_PH_MAP['Agent']:
                                            output.append((arg, 'Agent', {'cor tri cnt': a_cnt}))
                                    for arg in person:
                                        if arg != ROLE_PH_MAP['Person']:
                                            output.append((arg, 'Person', {'cor tri cnt': a_cnt}))
                                    for arg in destination:
                                        if arg != ROLE_PH_MAP['Destination']:
                                            output.append((arg, 'Destination', {'cor tri cnt': a_cnt}))          
                                    for arg in origin:
                                        if arg != ROLE_PH_MAP['Origin']:
                                            output.append((arg, 'Origin', {'cor tri cnt': a_cnt}))
                            except:
                                pass
                        used_o_cnt += 1
        return output

class Justice_Acquit(event_template):
    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)

    @classmethod
    def get_keywords(self):
        return ['acquitted', 'acquittal', 'acquit']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
                if o_style == 'argument:sentence':
                    output_template += ' \n {}'.format('somebody was acquitted of the charges by some adjudicator.')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('justice event, acquit sub-type')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format("The event is related to someone being acquitted.")
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
                if o_style == 'argument:sentence':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            " and ".join([ a['argument text'] for a in argu['Defendant']]) if "Defendant" in argu.keys() else ROLE_PH_MAP['Defendant'],
                            " and ".join([ a['argument text'] for a in argu['Adjudicator']]) if "Adjudicator" in argu.keys() else ROLE_PH_MAP['Adjudicator']
                        )
                        output_texts.append("{} was acquitted of the charges by {}.".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
                    if o_style == 'argument:sentence':
                        if used_o_cnt == cnt:
                            try:
                                for a_cnt, prediction in enumerate(full_pred.split(' <sep> ')):
                                    defendant = prediction.split(' was acquitted of the charges by ', 1)[0]
                                    defendant = defendant.split(' and ')
                                    remain = prediction.split(' was acquitted of the charges by ', 1)[1]

                                    adjudicator = remain.rsplit('.',1)[0]
                                    adjudicator = adjudicator.split(' and ')
                                    for arg in defendant:
                                        if arg != ROLE_PH_MAP['Defendant']:
                                            output.append((arg, 'Defendant', {'cor tri cnt': a_cnt}))
                                    for arg in adjudicator:
                                        if arg != ROLE_PH_MAP['Adjudicator']:
                                            output.append((arg, 'Adjudicator', {'cor tri cnt': a_cnt}))
                            except:
                                pass
                        used_o_cnt += 1
        return output

class Justice_Pardon(event_template):
    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)

    @classmethod
    def get_keywords(self):
        return ['pardon', 'pardoned', 'remission']

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
                if o_style == 'argument:sentence':
                    output_template += ' \n {}'.format('somebody received a pardon from some adjudicator.')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('justice event, pardon sub-type')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format("The event is related to someone being pardoned.")
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
                if o_style == 'argument:sentence':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            " and ".join([ a['argument text'] for a in argu['Defendant']]) if "Defendant" in argu.keys() else ROLE_PH_MAP['Defendant'],
                            " and ".join([ a['argument text'] for a in argu['Adjudicator']]) if "Adjudicator" in argu.keys() else ROLE_PH_MAP['Adjudicator']
                        )
                        output_texts.append("{} received a pardon from {}.".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
                    if o_style == 'argument:sentence':
                        if used_o_cnt == cnt:
                            try:
                                for a_cnt, prediction in enumerate(full_pred.split(' <sep> ')):
                                    defendant = prediction.split(' received a pardon from ', 1)[0]
                                    defendant = defendant.split(' and ')
                                    remain = prediction.split(' received a pardon from ', 1)[1]

                                    adjudicator = remain.rsplit('.',1)[0]
                                    adjudicator = adjudicator.split(' and ')
                                    for arg in defendant:
                                        if arg != ROLE_PH_MAP['Defendant']:
                                            output.append((arg, 'Defendant', {'cor tri cnt': a_cnt}))
                                    for arg in adjudicator:
                                        if arg != ROLE_PH_MAP['Adjudicator']:
                                            output.append((arg, 'Adjudicator', {'cor tri cnt': a_cnt}))
                            except:
                                pass
                        used_o_cnt += 1
        return output

class Justice_Appeal(event_template):
    def __init__(self, input_style, output_style, passage, event_type, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, gold_event)
    
    @classmethod
    def get_keywords(self):
        return ['appeal', 'appealing', 'appeals']  

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    output_template += ' \n {}'.format('Event trigger is <Trigger>')
                if o_style == 'argument:sentence':
                    output_template += ' \n {}'.format('some other in somewhere appealed the adjudication from some adjudicator.')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'event_type':
                    input_str += ' \n {}'.format('justice event, appeal sub-type')
                if i_style == 'event_type_sent':
                    input_str += ' \n {}'.format("The event is related to someone appealing the decision of a court.")
                if i_style == 'keywords':
                    input_str += ' \n Similar triggers such as {}'.format(', '.join(self.get_keywords()))
                if i_style == 'triggers':
                    input_str += ' \n The event trigger word is {}'.format(query_trigger)
                if i_style == 'template':
                    input_str += ' \n {}'.format(self.output_template)
        return input_str

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'trigger:sentence':
                    if self.trigger_text != '':
                        output_str += ' \n Event trigger is {}'.format(self.trigger_text)
                        gold_sample = True
                    else:
                        output_str += ' \n Event trigger is <Trigger>'
                if o_style == 'argument:sentence':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            " and ".join([ a['argument text'] for a in argu['Plaintiff']]) if "Plaintiff" in argu.keys() else ROLE_PH_MAP['Plaintiff'],
                            " and ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else ROLE_PH_MAP['Place'],
                            " and ".join([ a['argument text'] for a in argu['Adjudicator']]) if "Adjudicator" in argu.keys() else ROLE_PH_MAP['Adjudicator']
                        )
                        output_texts.append("{} in {} appealed the adjudication from {}.".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'trigger:sentence':
                        if used_o_cnt == cnt:
                            try:
                                triggers = full_pred.split('Event trigger is ', 1)[1]
                                triggers = triggers.split(' and ')
                                for t_cnt, t in enumerate(triggers):
                                    if t != '<Trigger>':
                                        output.append((t, self.event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                            except:
                                pass
                        used_o_cnt += 1
                    if o_style == 'argument:sentence':
                        if used_o_cnt == cnt:
                            try:
                                for a_cnt, prediction in enumerate(full_pred.split(' <sep> ')):
                                    plaintiff = prediction.split(' in ', 1)[0]
                                    plaintiff = plaintiff.split(' and ')
                                    remain = prediction.split(' in ', 1)[1]

                                    place = remain.split(' appealed the adjudication from ', 1)[0]
                                    place = place.split(' and ')
                                    remain = remain.split(' appealed the adjudication from ', 1)[1]

                                    adjudicator = remain.rsplit('.',1)[0]
                                    adjudicator = adjudicator.split(' and ')

                                    for arg in plaintiff:
                                        if arg != ROLE_PH_MAP['Plaintiff']:
                                            output.append((arg, 'Plaintiff', {'cor tri cnt': a_cnt}))
                                    for arg in place:
                                        if arg != ROLE_PH_MAP['Place']:
                                            output.append((arg, 'Place', {'cor tri cnt': a_cnt}))
                                    for arg in adjudicator:
                                        if arg != ROLE_PH_MAP['Adjudicator']:
                                            output.append((arg, 'Adjudicator', {'cor tri cnt': a_cnt}))
                            except:
                                pass
                        used_o_cnt += 1
        return output