from . import units
from .dssm_preprocessor import DSSMPreprocessor
from .naive_preprocessor import NaivePreprocessor
from .basic_preprocessor import BasicPreprocessor
from .cdssm_preprocessor import CDSSMPreprocessor
# from .mz_pretrained_preprocessor import PreTrainedModelsProcessor
# from .char_ngram_preprocessor import CharNGramPreprocessor
from .elmo_basic_preprocessor import ElmoPreprocessor
# from .bow_preprocessor import BoWPreprocessor
# from .declare_preprocessor import DeClarePreprocessor
# from .fact_checking_elmo_preprocessor import FactCheckingElmoPreprocessor
# from .char_man_preprocessor import CharManPreprocessor
# from .char_man_elmo_preprocessor import CharManElmoPreprocessor


def list_available() -> list:
    from matchzoo.engine.base_preprocessor import BasePreprocessor
    from matchzoo.utils import list_recursive_concrete_subclasses
    return list_recursive_concrete_subclasses(BasePreprocessor)
