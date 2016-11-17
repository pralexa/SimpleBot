
import abc


class Dataset(object):

    __metaclass__ = abc.ABCMeta

    @staticmethod
    @abc.abstractmethod
    def prepare_data(vocabulary_size, data_dir, tokenizer=None):
        return