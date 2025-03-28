from ..tokenizer import NormalizedString
class Normalizer:
    """
    Base class for all normalizers

    This class is not supposed to be instantiated directly. Instead, any implementation of a
    Normalizer will return an instance of this class when instantiated.
    """
    def normalize(self, normalized: NormalizedString):
        """
        Normalize a :class:`~tokenizers.NormalizedString` in-place

        This method allows to modify a :class:`~tokenizers.NormalizedString` to
        keep track of the alignment information. If you just want to see the result
        of the normalization on a raw string, you can use
        :meth:`~tokenizers.normalizers.Normalizer.normalize_str`

        Args:
            normalized (:class:`~tokenizers.NormalizedString`):
                The normalized string on which to apply this
                :class:`~tokenizers.normalizers.Normalizer`
        """
        pass

    def normalize_str(self, sequence: str):
        """
        Normalize the given string

        This method provides a way to visualize the effect of a
        :class:`~tokenizers.normalizers.Normalizer` but it does not keep track of the alignment
        information. If you need to get/convert offsets, you can use
        :meth:`~tokenizers.normalizers.Normalizer.normalize`

        Args:
            sequence (:obj:`str`):
                A string to normalize

        Returns:
            :obj:`str`: A string after normalization
        """
        pass