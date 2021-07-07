import argparse
import load_models

list_algorithms = [
    'erm',
    'arm',
    'fish',
    'cdann',
    'g2dm',
    'iga',
    'irm'
]


for backbone in ['r2p1d', 'i3d', 'stam', 'x3d']:
    for pre_extract_flow in [True, False]:
        for algorithm_mode in list_algorithms:
            print()
