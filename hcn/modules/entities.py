from collections import defaultdict

import numpy as np


class EntityTracker(object):
    def __init__(self, in_kb):
        self.slot_value_mapping = in_kb
        self.ordered_kb = self.make_ordered_kb(in_kb)
        self.num_features = len(self.slot_value_mapping)
        self.entities = {key: None for key in self.slot_value_mapping.keys()}

    @staticmethod
    def make_ordered_kb(in_kb):
        ordered_kb = []
        for key, values in in_kb.items():
            ordered_kb += [(value, key) for value in values]
        return sorted(ordered_kb, key=lambda x: len(x[0]), reverse=True)

    def reset(self):
        self.entities = {key: None for key in self.slot_value_mapping.keys()}

    @staticmethod
    def read_kb(in_file):
        entity_map = defaultdict(lambda: set([]))

        with open(in_file, encoding='utf-8') as kb_in:
            for line in kb_in:
                idx, restaurant_name, feature, value = line.lower().split()
                entity_map[feature].add(value)
                entity_map['r_name'].add(restaurant_name)
        return entity_map

    def extract_entities(self, utterance, update=True):
        utterance_delexicalized = utterance[:]
        for value, key in self.ordered_kb:
            if value in utterance_delexicalized:
                utterance_delexicalized = utterance_delexicalized.replace(value, key)
                if update:
                    self.entities[key] = True
        return utterance_delexicalized

    def context_features(self):
        keys = sorted(set(self.entities.keys()))
        ctxt_features = np.array([bool(self.entities[key]) for key in keys], dtype=np.float32)
        return ctxt_features

    def action_mask(self):
        raise NotImplementedError('Not yet implemented. Need a list of action templates!')
