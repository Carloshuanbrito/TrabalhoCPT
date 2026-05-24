#!/usr/bin/env python3
"""Extract code cells from Jupyter notebook."""

import json
import sys

def extract_code_cells(notebook_path):
    """Extract all code cells from a Jupyter notebook."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            print(f"\n{'='*80}")
            print(f"CELL {idx}:")
            print('='*80)
            print(source)

if __name__ == '__main__':
    extract_code_cells('matrix_analysis2.ipynb')
