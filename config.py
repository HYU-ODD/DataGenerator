import sys
import yaml
import glob
import os
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


def cfg_from_yaml_file(cfg_file):
    with open(cfg_file, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)

    return config

def config_to_trans(trans_config):
    transform = carla.Transform(carla.Location(trans_config["location"][0],
                                               trans_config["location"][1],
                                               trans_config["location"][2]),
                                carla.Rotation(trans_config["rotation"][0],
                                               trans_config["rotation"][1],
                                               trans_config["rotation"][2]))
    return transform