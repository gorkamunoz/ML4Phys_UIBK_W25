import sys
import torch
import os


def ingestion(path_to_model, # path to where your model is
              input_dir, # path to the folder where the training data
              output_dir # path where your predictions will be saved
             ):
    sys.path.append(path_to_model)
    from model import model

    # ========= Load data
    print("======== Loading data ==========")
    samples = torch.load(os.path.join(input_dir, 'input_data.pt'))

    # ========= Creating a model
    print("======== Creating model ==========")
    M = model()
    preds = M.pred(samples)    
    
    # ========= Write results
    print("======== Writing results ==========")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    torch.save(preds, os.path.join(output_dir, 'preds.pt'))

def scoring(truth_dir, # path to the folder where the training data is
           submit_dir # path where your predictions were saved (i.e. output_dir from ingestion)
              ):
    
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
    
    return loss