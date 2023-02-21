import json
import os.path
import sys
import subprocess
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

with open("test_spike.json", 'r') as f:
    data = json.load(f)
    path = os.path.join(ROOT_DIR, data['path'])
    config = os.path.join(ROOT_DIR, data['config'])
    checkpoint = os.path.join(ROOT_DIR, data['checkpoint'])
    cfg_keys = list(data.keys())[3:]
    cfg_values = list(data.values())[3:]
    f.close()

sh = f'python {path} {config} {checkpoint}'
for key, value in zip(cfg_keys, cfg_values):
    if key == 'cfg-options':
    	sh = sh + ' ' + f'--{key}'
    	for k, v in data[key].items():
        	sh = sh + ' ' + f'{k}={v}'
    elif value == "True":
        sh = sh + ' ' f'--{key}'
    elif isinstance(value, float):
        sh = sh + ' ' f'--{key}={value}'
    else:
        sh = sh + ' ' + f'--{key} {value}'
    

# print(sh)
p = subprocess.Popen(sh, shell=True)
p.communicate()
sys.exit()
