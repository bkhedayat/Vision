import os

def check_file(file):
    # checks if the file exists
    # convert file to str format
    file = str(file)
    if os.path.isfile(file):
        return file