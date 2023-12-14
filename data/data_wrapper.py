from ..Utils.utils import parse_yaml

class DataWrapper():
    """
    dataset wrapper
    recieves data yaml file
    """

    def __init__(self, data_yaml):
        self.hyper = parse_yaml(data_yaml)
        self.class_names = self.hyper['class_names']

    def create_datasets(self, cls_num):
        """
        Creates train, valid and test sets
        """
        
        # check the class number with len of class names
        assert len(self.class_names) == cls_num, "create_datasets: ERROR: class names & number of classes are different!"

