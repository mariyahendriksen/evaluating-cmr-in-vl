from src.perturbations.perturbation_types.typos import TyposPerturbation
from src.perturbations.perturbation_types.synonym_based import SynonymBased
from src.perturbations.perturbation_types.distraction_based import DistractionBased
from src.perturbations.perturbation_types.ARO import ARO


PERTURBATION_TYPES = (
    'typos',
    'synonym_noun',
    'distraction_true',
    'shuffle_nouns_and_adj'

)

TYPOS_PERTURBATION_TYPES = (
    'char_swap',
    'missing_char',
    'extra_char',
    'nearby_char',
    'probability_based_letter_change'
)

SYNONYM_PERTURBATION_TYPES = (
    'synonym_noun',
    'synonym_adj'
)

DISTRACTION_PERTURBATION_TYPES = (
    'distraction_true',
    'distraction_false'
)

ARO_PERTURBATION_TYPES = (
    'shuffle_nouns_and_adj',
    'shuffle_all_words',
    'shuffle_allbut_nouns_and_adj',
    'shuffle_within_trigrams',
    'shuffle_trigrams'
)

class Perturbation(object):

    def __init__(self, config, perturbation_type) -> None:

        super(Perturbation, self).__init__()

        self.config = config
        self.perturbation_type = perturbation_type
        self.perturbation = self.get_perturbation()
        # print('Initialized perturation: ', self.perturbation_type)

    def __repr__(self) -> str:
        return f'Perturbation type: {self.config.args.perturbation}'

    def get_perturbation(self):

        perturbation_type = self.perturbation_type
        # print('perturbation_type: ', perturbation_type)

        if perturbation_type in TYPOS_PERTURBATION_TYPES:
            perturbation = TyposPerturbation(type=perturbation_type)
        elif perturbation_type in SYNONYM_PERTURBATION_TYPES:
            perturbation = SynonymBased(type=perturbation_type)
        elif perturbation_type in DISTRACTION_PERTURBATION_TYPES:
            perturbation = DistractionBased(type=perturbation_type)
        elif perturbation_type in ARO_PERTURBATION_TYPES:
            perturbation = ARO(type=perturbation_type)
        else:
            raise NotImplementedError
        
        return perturbation
    
    def apply_perturbation_to_caption(self, caption):
        return self.perturbation.apply_perturbation_to_caption(caption=caption)
