with open('requirements.txt', 'r') as f:
    data = f.read()
    f.close()

excludes = ['nvidia', 'torch', 'numpy', 'mm', 'wandb']

new_data = f''

for i, row in enumerate(data.split('\n')):
    package = row.split('==')[0]
    if any(exclude in package for exclude in excludes):
        package = row
    new_data = new_data + f'{package}\n'

with open('requirements.txt', 'w') as f:
    f.write(new_data)