import json
with open(r'D:\Study\cifar100\train_colab_vit.ipynb', 'r', encoding='utf-8') as f:
    data = json.load(f)
for i, c in enumerate(data['cells']):
    src = ' '.join(c['source'])[:120]
    print(f"Cell {i} ({c['cell_type']}): {src}")
print(f"\nTotal cells: {len(data['cells'])}")
