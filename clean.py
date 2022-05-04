import os
import json
import shutil
from pathlib import Path

for i in range(3):
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
