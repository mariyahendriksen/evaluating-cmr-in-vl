from typing import Any
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize


SYNONYM_PERTURBATION_TYPES = (
    'synonym_noun',
    'synonym_adj'
)


class SynonymBased:

    def __init__(self, type='synonym_noun') -> None:

        self.tokenizer = word_tokenize
        self.type = type

    def get_synonyms(self, seed_word):
        # print(seed_word)
        synonyms = []
        for syn in wordnet.synsets(seed_word):
            for l in syn.lemmas():
                # print('l name: ', l.name())
                # print(f'{seed_word} != {l.name()}?', seed_word != l.name() )
                lemma_name = l.name()
                if lemma_name not in synonyms and lemma_name.lower() != seed_word.lower():
                    # print(f'addng "{l.name()}" to the list')
                    lemma_name = lemma_name.replace('_', ' ')
                    synonyms.append(lemma_name) 
        return synonyms

    def replace_nouns_with_synonyms(self, caption, num_of_words_to_replace=1):

        new_caption = []
        counter = 0
        # identify nouns
        tokens = self.tokenizer(caption)
        for word, pos in nltk.pos_tag(tokens):
            if pos in ['NN', 'NNS', 'NNP', 'NNPS'] and counter < num_of_words_to_replace:
                synonyms = self.get_synonyms(seed_word=word)
                if len(synonyms) > 0:
                    new_noun = synonyms[0]
                    new_caption.append(new_noun)
                    counter += 1
                else:
                    new_caption.append(word)
            else:
                new_caption.append(word)

        return ' '.join(new_caption)


    def replace_adj_with_synonyms(self, caption, num_of_words_to_replace=1):

        new_caption = []
        counter = 0
        # identify adj
        tokens = self.tokenizer(caption)
        for word, pos in nltk.pos_tag(tokens):
            if pos in ['JJ', 'JJR', 'JJS'] and counter < num_of_words_to_replace:
                synonyms = self.get_synonyms(seed_word=word)
                if len(synonyms) > 0:
                    new_noun = synonyms[0]
                    new_caption.append(new_noun)
                    counter += 1
                else:
                    new_caption.append(word)
            else:
                new_caption.append(word)

        return ' '.join(new_caption)

    def apply_perturbation_to_caption(self, caption: str) -> str:
        if self.type == 'synonym_noun':
            return self.replace_nouns_with_synonyms(
                caption=caption,
                num_of_words_to_replace=1
                )
        elif self.type == 'synonym_adj':
            return self.replace_adj_with_synonyms(
                caption=caption,
                num_of_words_to_replace=1
                )
        else:
            raise NotImplementedError
