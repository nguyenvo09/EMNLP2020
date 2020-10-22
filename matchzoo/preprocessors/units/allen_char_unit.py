from .unit import Unit
from allennlp.modules.elmo import batch_to_ids


class AllenCharUnit(Unit):
    """
    Process unit for n-letter generation.

    Triletter is used in :class:`DSSMModel`.
    This processor is expected to execute before `Vocab`
    has been created.

    Examples:
        >>> triletter = AllenCharUnit()
        >>> rv = triletter.transform(['hello', 'word'])
        >>> len(rv)
        9
        >>> rv
        ['#he', 'hel', 'ell', 'llo', 'lo#', '#wo', 'wor', 'ord', 'rd#']
        >>> triletter = AllenCharUnit(reduce_dim=False)
        >>> rv = triletter.transform(['hello', 'word'])
        >>> len(rv)
        2
        >>> rv
        [['#he', 'hel', 'ell', 'llo', 'lo#'], ['#wo', 'wor', 'ord', 'rd#']]

    """

    def __init__(self, max_len: int, pad_mode='post'):
        """
        Class initialization.

        :param ngram: By default use 3-gram (tri-letter).
        :param reduce_dim: Reduce to 1-D list for sentence representation.
        """
        self._max_len = max_len
        self._pad_mode = pad_mode

    def transform(self, input_: list) -> list:
        """
        Transform token into letter.

        For example, `word` should be represented as `w`, `o`, `r`, `d`

        :param input_: list of tokens to be transformed.

        :return n_letters: generated n_letters.
        """
        fixed_tokens = [[0] * 50 for _ in range(self._max_len)]
        input_ = [input_[:self._max_len]]  # to satisfy the f*** function
        ans = batch_to_ids(input_)[0].tolist()  # List[List[int]] len <= self.max_len, each element is 50 numbers.
        # padding 0 to ans
        if self._pad_mode == 'post':
            fixed_tokens[:len(ans)] = ans
        elif self._pad_mode == 'pre':
            fixed_tokens[-len(ans):] = ans
        else:
            raise ValueError('{} is not a vaild '
                             'pad mode.'.format(self._pad_mode))
        return fixed_tokens
