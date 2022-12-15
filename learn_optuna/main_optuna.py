##
import torch
import numpy as np
import pandas as pd
import os
import pickle
import random
from envs.train import data_preprocessing_CycleGAN,data_preprocessing_Combat, partition, partition_kfold, optuna_tuning, experiment, experiment_kfold
from config import load_arguments


## ========= Run Experiment and saving result ========= ##
if __name__ == '__main__':
    args = load_arguments()
   
    # seed number for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    ## make folders
    os.makedirs(args.result_dir,exist_ok=True)
    os.makedirs(args.ckpt_dir,exist_ok=True)
    os.makedirs(args.log_dir,exist_ok=True)
    
    # Data preprocessing
    #imageFiles_labels = data_preprocessing_CycleGAN(args)
    imageFiles_labels = data_preprocessing_Combat(args)
    
    if args.k_folds == 1: 
        # Data split
        part = partition(imageFiles_labels,args)
       
        # Run Experiment
        # load best arguments
        if os.path.exists(os.path.join(args.log_dir,f'best_args_{args.study_name}.pkl')):
            with open(os.path.join(args.log_dir,f'best_args_{args.study_name}.pkl'),'rb') as f:
                print('loading best arguments')
                best_args = pickle.load(f)
                for key,value in best_args.items():
                    setattr(args,key,value)
                best_args = args
        else:
            # optuna tuning
            best_args = optuna_tuning(part, args)
            # save best arguments
            with open(os.path.join(args.log_dir,f'best_args_{args.study_name}.pkl'),'wb') as f:
                print('saving best arguments')
                pickle.dump(vars(best_args),f)


        # train with best params and test
        final_args, result = experiment(part,best_args)
        print(f"test_acc_or_R^2 is {result['test_acc_or_R^2']}")
        print(f"test_auc_or_test_loss is {result['test_auc_or_test_loss']}")
        
        # save the results
        final_result = pd.DataFrame({'test_acc_or_R^2': [result['test_acc_or_R^2']], 'test_auc_or_test_loss' : [result['test_auc_or_test_loss']]})
        final_result.to_csv(os.path.join(args.result_dir, 'result.csv'))
        
    
    # kfold
    elif args.k_folds > 1:
        partitions = partition_kfold(imageFiles_labels,args)
        final_args, result = experiment_kfold(partition,args) 
    
## ==================================================== ##
