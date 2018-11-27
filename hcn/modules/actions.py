import numpy as np

from utils.preprocessing import PAD, UNK

'''
    Action Templates

    1. 'any preference on a type of cuisine',
    2. 'api_call <party_size> <rest_type>',
    3. 'great let me do the reservation',
    4. 'hello what can i help you with today',
    5. 'here it is <info_address>',
    6. 'here it is <info_phone>',
    7. 'how many people would be in your party',
    8. "i'm on it",
    9. 'is there anything i can help you with',
    10. 'ok let me look into some options for you',
    11. 'sure is there anything else to update',
    12. 'sure let me find an other option for you',
    13. 'what do you think of this option: ',
    14. 'where should it be',
    15. 'which price range are looking for',
    16. "you're welcome",

    [1] : cuisine
    [2] : location
    [3] : party_size
    [4] : rest_type

'''


class ActionTracker(object):
    def __init__(self, in_candidates_file, ent_tracker, action_templates=None):
        # maintain an instance of EntityTracker
        self.et = ent_tracker
        # get a list of action templates
        if not action_templates:
            action_templates = self.load_action_templates(in_candidates_file) \
                if in_candidates_file \
                else []
        if UNK not in action_templates:
            action_templates.append(UNK)
        if PAD not in action_templates:
            action_templates.append(PAD)
        self.set_action_templates(action_templates)

    def action_mask(self):
        # get context features as string of ints (0/1)
        ctxt_f = ''.join([str(flag) for flag in self.et.context_features().astype(np.int32)])

        def construct_mask(ctxt_f):
            indices = self.am_dict['default']
            for index in indices:
                self.am[index-1] = 1.
            return self.am
    
        return construct_mask(ctxt_f)

    def load_action_templates(self, in_file):
        templates = set([])
        with open(in_file, encoding='utf-8') as templates_in:
            for line in templates_in:
                templates.add(line.strip().lower().partition(' ')[2])
        responses = [PAD, UNK] + sorted(set([self.et.extract_entities(response, update=False)
                                        for response in templates]))
        return responses

    def set_action_templates(self, in_action_templates):
        self.action_templates = in_action_templates
        self.action_size = len(self.action_templates)
        self.am = np.zeros([self.action_size], dtype=np.float32)
        self.am_dict = {'default': range(len(self.action_templates))}

