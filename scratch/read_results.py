import json
import os

path = r'c:\Users\MD.ZAID SHAIKH\Documents\Cognix\Cognix_07_09_2026\results\universal\main_results.json'
if os.path.exists(path):
    with open(path, 'r') as f:
        data = json.load(f)
        for k, v in data.items():
            if 'NORMAL' in k:
                acc = v.get('accuracy', 0)
                ece = v.get('ece', 0)
                cov = v.get('conformal_coverage', 0)
                print(f'{k}: ACC={acc:.4f}, ECE={ece:.4f}, Cov={cov:.4f}')
else:
    print('No results file found.')
