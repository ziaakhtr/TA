import json
with open('playground.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
print('Valid JSON')
for i, c in enumerate(nb['cells']):
    print(f'Cell {i}: id={c["id"]}')
