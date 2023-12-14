from ..Utils.utils import parse_yaml

class DataWrapper():
    """
    dataset wrapper
    recieves data yaml file
    """

    def __init__(self, data_yaml):
        self.hyper = parse_yaml(data_yaml)

    def create_datasets():
        """
        Creates train, valid and test sets
        """
        pass