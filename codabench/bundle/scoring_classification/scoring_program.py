import os
import sys
import torch

    

print("======== Entered classification scoring function ========")  

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
# Load prediction file
print("======== Loading predicted labels ==========")
preds = torch.load(os.path.join(submit_dir, 'preds.pt'))

assert trues.shape == preds.shape, "Shape of predicted values are incorrect."

print("======== Calculating binary cross entropy loss ==========")
loss_func = torch.nn.BCELoss()
loss = loss_func(preds.to(float), trues.to(float))

print("======== Saving data ==========")
output_file.write(f'loss: {loss.item()}\n')



