
DISTRACTION_PERTURBATION_TYPES = (
    'distraction_true',
    'distraction_false'
)

class DistractionBased:

    def __init__(self, type='distraction_true') -> None:

        self.type = type
    
        self.true_statements = [
            'and true it true',
            'and false is false',
            'and false is not true'
        ]

    def add_true_is_true(self, caption) -> str:
        statement_to_append = 'and true it true'
        new_caption = caption.strip('.') + ' ' + statement_to_append + '.'
        return new_caption

    def add_false_is_false(self, caption):
        statement_to_append = 'and false is false'
        new_caption = caption.strip('.') + ' ' + statement_to_append + '.'
        return new_caption

    def add_url(self, caption):
        raise NotImplementedError

    def apply_perturbation_to_caption(self, caption: str) -> str:
        if self.type == 'distraction_true':
            return self.add_true_is_true(caption=caption)
        elif self.type == 'distraction_false':
            return self.add_false_is_false(caption=caption)
        else:
            raise NotImplementedError
