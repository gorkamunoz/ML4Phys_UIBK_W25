import os
import sys
import numpy as np


def regression_scoring(INPUT_DIR = None, # directory to where to find the reference and predicted labes
                           OUTPUT_DIR = None # directory where the scores will be saved (scores.txt)
                        ):
    
    if INPUT_DIR is None:
        INPUT_DIR = sys.argv[1]
    if OUTPUT_DIR is None:
        OUTPUT_DIR = sys.argv[2]
        
    submit_dir = os.path.join(INPUT_DIR, 'res')
    truth_dir = os.path.join(INPUT_DIR, 'ref')
    
    if not os.path.isdir(submit_dir):
        print( "%s doesn't exist", submit_dir)
        
    if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

    output_filename = os.path.join(OUTPUT_DIR, 'scores.txt')
    output_file = open(output_filename, 'w')

    # Load ground truth
    trues = np.load(os.path.join(truth_dir, 'true_reg.npy'))
    # Results file
    preds = np.load(os.path.join(submit_dir, 'preds.npy'))

    loss = np.mean((trues - preds) ** 2)

    output_file.write(f'loss: {loss}\n')