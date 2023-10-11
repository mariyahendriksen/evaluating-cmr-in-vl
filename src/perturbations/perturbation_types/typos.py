import random
import string
import typo


TYPOS_PERTURBATION_TYPES = (
    'char_swap',
    'missing_char',
    'extra_char',
    'nearby_char',
    'probability_based_letter_change'
)


class TyposPerturbation:

    def __init__(self, type='char_swap', seed=42) -> None:

        self.type = type
        assert type in TYPOS_PERTURBATION_TYPES, print('Uknown perturnation type: ', type)
        self.seed = seed

    def missing_char(self, caption):
        string = typo.StrErrer(caption, seed=self.seed)
        return string.missing_char().result 

    def char_swap(self, caption):
        string = typo.StrErrer(caption, seed=self.seed)
        return string.char_swap().result 

    def extra_char(self, caption):
        string = typo.StrErrer(caption, seed=self.seed)
        return string.extra_char().result 
    
    def nearby_char(self, caption):
        string = typo.StrErrer(caption, seed=self.seed)
        return string.nearby_char().result 

    def probability_based_letter_change(self, caption, probability_of_word_change=0.1):
        random.seed(self.seed)
        new_caption = []
        words = caption.split(' ')
        for word in words:
            outcome = random.random()
            if outcome <= probability_of_word_change:
                ix = random.choice(range(len(word)))
                new_word = ''.join([word[w] if w != ix else random.choice(string.ascii_letters) for w in range(len(word))])
                new_caption.append(new_word)
            else:
                new_caption.append(word)

        new_caption = ' '.join([w for w in new_caption])

        return new_caption

    def apply_perturbation_to_caption(self, caption: str) -> str:
        if self.type == 'char_swap':
            return self.char_swap(caption=caption)
        elif self.type == 'missing_char':
            return self.missing_char(caption=caption)
        elif self.type == 'extra_char':
            return self.extra_char(caption=caption)
        elif self.type == 'nearby_char':
            return self.nearby_char(caption=caption)
        elif self.type == 'probability_based_letter_change':
            return self.probability_based_letter_change(caption=caption)
        else:
            raise NotImplementedError
