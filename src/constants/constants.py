MODELS=[
    "altclip",
    "align",
    "clip",
    "groupvit"
]

DATASETS=[
    "coco",
    "coco_aug",
    "f30k",
    "f30k_aug"
]

TASKS = [
    'i2t',
    't2i'
]

PERTURBATIONS=[
    "none",
    'char_swap',
    'missing_char',
    'extra_char',
    'nearby_char',
    'probability_based_letter_change',
    "synonym_noun",
    'synonym_adj',
    "distraction_true",
    "distraction_false",
    "shuffle_nouns_and_adj",
    'shuffle_all_words',
    'shuffle_allbut_nouns_and_adj',
    'shuffle_within_trigrams',
    'shuffle_trigrams'
]
