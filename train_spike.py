import json
import os.path
import sys
import subprocess
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

with open("train_spike.json", 'r') as f:
    data = json.load(f)
    path = os.path.join(ROOT_DIR, data['path'])
    config = os.path.join(ROOT_DIR, data['config'])
    cfg_options = data['cfg-options']
    f.close()
if any(cfg_options):
    sh = f'python {path} {config} --cfg-options'
    for key, value in cfg_options.items():
        sh = sh + ' ' + f'{key}={value}'
else:
    sh = f'python {path} {config}'
# print(sh)
p = subprocess.Popen(sh, shell=True)
p.communicate()
sys.exit()
