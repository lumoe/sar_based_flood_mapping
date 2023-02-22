from yaml import load, FullLoader
from munch import Munch
   

def load_config():
    with open('config/config.yaml') as f:
        return load(f, Loader=FullLoader)
    

config = Munch.fromDict(load_config())