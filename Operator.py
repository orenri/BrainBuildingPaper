import abc


class Operator(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def operate(self, connectome_dev):
        ...
