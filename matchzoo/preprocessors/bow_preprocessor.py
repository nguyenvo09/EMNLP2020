"""Basic Preprocessor."""

from tqdm import tqdm

from . import units
from matchzoo import DataPack
from matchzoo.engine.base_preprocessor import BasePreprocessor
from .build_vocab_unit import build_vocab_unit
from .build_unit_from_data_pack import build_unit_from_data_pack
from .chain_transform import chain_transform
from handlers.output_handler import FileHandler
from typing import List
import torch
import itertools, os
from matchzoo.preprocessors.units import Unit

tqdm.pandas()


class BoWPreprocessor(BasePreprocessor):
    """
    Bag of word preprocessor. Fit is same as Basic Processor but transform will transform text into bag of words

    :param fixed_length_left: Integer, maximize length of :attr:`left` in the
        data_pack.
    :param fixed_length_right: Integer, maximize length of :attr:`right` in the
        data_pack.
    :param filter_mode: String, mode used by :class:`FrequenceFilterUnit`, Can
        be 'df', 'cf', and 'idf'.
    :param filter_low_freq: Float, lower bound value used by
        :class:`FrequenceFilterUnit`.
    :param filter_high_freq: Float, upper bound value used by
        :class:`FrequenceFilterUnit`.
    :param remove_stop_words: Bool, use :class:`StopRemovalUnit` unit or not.

    Example:
        >>> import matchzoo as mz
        >>> train_data = mz.datasets.toy.load_data('train')
        >>> test_data = mz.datasets.toy.load_data('test')
        >>> preprocessor = mz.preprocessors.BoWPreprocessor(
        ...     fixed_length_left=10,
        ...     fixed_length_right=20,
        ...     filter_mode='df',
        ...     filter_low_freq=2,
        ...     filter_high_freq=1000,
        ...     remove_stop_words=True
        ... )
        >>> preprocessor = preprocessor.fit(train_data, verbose=0)
        >>> preprocessor.context['input_shapes']
        [(10,), (20,)]
        >>> preprocessor.context['vocab_size']
        225
        >>> processed_train_data = preprocessor.transform(train_data,
        ...                                               verbose=0)
        >>> type(processed_train_data)
        <class 'matchzoo.data_pack.data_pack.DataPack'>
        >>> test_data_transformed = preprocessor.transform(test_data,
        ...                                                verbose=0)
        >>> type(test_data_transformed)
        <class 'matchzoo.data_pack.data_pack.DataPack'>

    """

    def __init__(self, fixed_length_left: int = 30,
                 fixed_length_right: int = 30,
                 filter_mode: str = 'df',
                 filter_low_freq: float = 2,
                 filter_high_freq: float = float('inf'),
                 remove_stop_words: bool = False,
                 right_visual_features_pth: str = None,
                 fixed_num_images_right: int = 1):
        """Initialization."""
        super().__init__()
        self._fixed_length_left = fixed_length_left
        self._fixed_length_right = fixed_length_right
        self._left_fixedlength_unit = units.FixedLength(
            self._fixed_length_left,
            pad_mode='post'
        )
        self._right_fixedlength_unit = units.FixedLength(
            self._fixed_length_right,
            pad_mode='post'
        )
        self._filter_unit = units.FrequencyFilter(
            low=filter_low_freq,
            high=filter_high_freq,
            mode=filter_mode
        )
        self._units = self._default_units()
        self._images_unit = ImagesUnit(right_visual_features_pth, fixed_num_images_right)
        if remove_stop_words:
            self._units.append(units.stop_removal.StopRemoval())

    def fit(self, data_pack: DataPack, verbose: int = 1):
        """
        Fit pre-processing context for transformation.

        :param data_pack: data_pack to be preprocessed.
        :param verbose: Verbosity.
        :return: class:`BasicPreprocessor` instance.
        """
        data_pack = data_pack.apply_on_text(chain_transform(self._units),
                                            verbose=verbose)
        fitted_filter_unit = build_unit_from_data_pack(self._filter_unit,
                                                       data_pack,
                                                       flatten=False,
                                                       mode='right',
                                                       verbose=verbose)
        data_pack = data_pack.apply_on_text(fitted_filter_unit.transform,
                                            mode='right', verbose=verbose)
        self._context['filter_unit'] = fitted_filter_unit

        vocab_unit = build_vocab_unit(data_pack, verbose=verbose, mode="right")  # only rely on the right side
        self._context['vocab_unit'] = vocab_unit

        vocab_size = len(vocab_unit.state['term_index'])  # + 1  # +1 for padding
        self._context['vocab_size'] = vocab_size
        self._context['embedding_input_dim'] = vocab_size
        self._context['input_shapes'] = [(self._fixed_length_left,),
                                         (self._fixed_length_right,)]

        return self

    def transform(self, data_pack: DataPack, verbose: int = 1) -> DataPack:
        """
        Apply transformation on data, create fixed length representation.

        :param data_pack: Inputs to be preprocessed.
        :param verbose: Verbosity.

        :return: Transformed data as :class:`DataPack` object.
        """
        data_pack = data_pack.copy()
        data_pack.apply_on_text(chain_transform(self._units), inplace=True,
                                verbose=verbose)

        data_pack.apply_on_text(self._context['filter_unit'].transform,
                                mode='right', inplace=True, verbose=verbose)

        def convert_to_bow(input_: List[str]):
            """the list of tokens will be converted to """
            vocab_unit = self._context['vocab_unit']
            ans = [0.0] * self._context['vocab_size']
            for token in input_:
                index = vocab_unit._state['term_index'][token]
                ans[index] = 1.0
            return ans

        data_pack.apply_on_text(convert_to_bow, mode='both', inplace=True, verbose=verbose)
        data_pack.right['images_right'] = data_pack.right["images_right"].progress_apply(self._images_unit.transform)
        return data_pack


class ImagesUnit(Unit):
    def __init__(self, visual_features_pth: str, max_len_images: int):
        """

        Parameters
        ----------
        visual_features_pth: str the path to pre-extracted features from images
        max_len_images: str the maxinum number of images used
        """
        self.fat_tensor, paths = torch.load(os.path.join("..", visual_features_pth))
        self.mapper = dict(zip(paths, range(len(paths))))  # indices here are rows of `fat_tensor`
        assert len(paths) == self.fat_tensor.size(0)  # ensure consistency
        self.pad = [0.0] * 4096  # for padding!!!
        self.max_len_images = max_len_images

    def transform(self, images: str) -> list:
        """
        Process input data from raw terms to list of tokens.

        :param images: a list of images path.

        :return tokens: tokenized tokens as a list.
        """
        images = [self.fat_tensor[self.mapper[p]].numpy().tolist() for p in images[:self.max_len_images]]
        if len(images) < self.max_len_images:  # padding
            images.extend([self.pad for _ in range(self.max_len_images - len(images))])
        return list(itertools.chain.from_iterable(images))