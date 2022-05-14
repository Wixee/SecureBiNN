import os
import json
import shutil
from pathlib import Path


config = None

with open('role/config.json', 'r') as f:
    config = json.load(f)


for i in range(3):
    name = None
    if i == 0:
        name = 'data_owner'
    elif i == 1:
        name = 'model_owner'
    else:
        name = 'ttp'

    new_dir_path = Path('role_{}_{}'.format(i, name))

    # clean the files
    if os.path.exists(new_dir_path):
        shutil.rmtree(new_dir_path)

    # copy the file
    shutil.copytree('role', new_dir_path)
    cur_config = config.copy()

    # modify
    cur_config['role'] = i

    cur_r_lst = config['r'].copy()
    cur_r_lst[(i-1) % 3] = -1
    cur_config['r'] = cur_r_lst
    cur_config['ip_{}'.format(i)] = '0.0.0.0'

    del cur_config['port_{}_{}'.format((i+1) % 3, (i-1) % 3)]
    del cur_config['port_{}_{}'.format((i-1) % 3, (i+1) % 3)]
    
    if i != 0:
        del cur_config['dataset']
        del cur_config['archive_path']
        if os.path.exists(new_dir_path.joinpath('archives')):
            shutil.rmtree(new_dir_path.joinpath('archives'))
    if i != 1:
        del cur_config['model_path']
        if os.path.exists(new_dir_path.joinpath('models')):
            shutil.rmtree(new_dir_path.joinpath('models'))

    new_config_path = new_dir_path.joinpath('config.json')
    with open(new_config_path, 'w') as f:
        json.dump(cur_config, f, indent=4)
