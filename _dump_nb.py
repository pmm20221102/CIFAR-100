import json
nb = json.load(open('train_local_vit.ipynb', 'r', encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    print(f'--- Cell {i} ({c["cell_type"]}) ---')
    print(''.join(c['source']))
    print()
