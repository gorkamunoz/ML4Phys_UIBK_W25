import sys
import os
import torch


print("======== Entered regression scoring function ========")  

INPUT_DIR = sys.argv[1]
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

print("======== Loading ground truth labels ==========")
trues = torch.load(os.path.join(truth_dir, 'trues.pt'))
# Results file
print("======== Loading predicted labels ==========")
preds = torch.load(os.path.join(submit_dir, 'preds.pt'))

assert trues.shape == preds.shape, "Shape of predicted values are incorrect."

print("======== Calculating MSE ==========")
loss = torch.mean((trues - preds) ** 2)

print("======== Saving data ==========")
output_file.write(f'loss: {loss}\n')