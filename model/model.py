import yaml


class YamlParser:
    # YamlParser parses yaml config file and create the model
    def __init__ (self, file_name):
        self.file_name = file_name
    
    def parse_config(self):
        with open(self.file_name) as f:
            model_config = yaml.load(f, Loader=yaml.FullLoader)
