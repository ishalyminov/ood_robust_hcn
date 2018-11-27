import re
from collections import defaultdict

SLOT_REGEXPS = {'r_cuisine': ['you are looking for a (\w+) restaurant right?'],
                'r_name': ['the post code of (\w+) is \w+',
                           '^(.+) is a great restaurant',
                           '(.+) is a good restaurant'],
                'r_address': ['^the address of .+ is (.+) .$',
                              'is a good restaurant on (.+) and'],
                'r_phone': ['their phone number is (.+) .'],
                'r_post_code': [', at (.+)$', 'and their post code is (.+)']}


def mine_custom_slot_values(in_candidates, in_kb):
    for utterance in in_candidates:
        for slot_name, regexps in SLOT_REGEXPS.items():
            for regex in regexps:
                in_kb[slot_name].update(re.findall(regex, utterance))


def make_augmented_knowledge_base(in_kb_file, in_candidates_file):
    kb = defaultdict(lambda: set([]))
    with open(in_kb_file, encoding='utf-8') as kb_in:
        for line in kb_in:
            idx, rest_name, key, value = line.lower().split()
            kb[key].add(value)
            kb['r_name'].add(rest_name)
    with open(in_candidates_file, encoding='utf-8') as cand_in:
        candidates = [line.strip().lower().partition(' ')[2] for line in cand_in]
    mine_custom_slot_values(candidates, kb)
    return {slot: list(values) for slot, values in kb.items()}
