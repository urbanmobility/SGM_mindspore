import time
import copy
import numpy as np

import mindspore as ms
from mindspore import nn
from mindspore import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor, Callback
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
import pandas as pd
from tqdm import tqdm
from mindspore import save_checkpoint, load_checkpoint, load_param_into_net
from src.SGAE import SGAE
from dataloader import LoadDocumentData, LoadImageData, LoadTabularData
from mindspore import context, Tensor
from mindspore import dtype as mstype
from dataset import create_dataset
import os
import time
import argparse
import pandas as pd
from model_utils.config import config as cfg


def eval_tabular(params):

    if params.ms_mode=="GRAPH":
        context.set_context(mode=context.GRAPH_MODE)
    else:
        context.set_context(mode=context.PYNATIVE_MODE,pynative_synchronize=True)
    context.set_context(device_target=params.device)
    # Load data
    x_train, y_train, x_val, y_val, x_test, y_test = LoadTabularData(params)
    x_train_whole=Tensor(x_train,dtype=mstype.float32)
    x_test_whole=Tensor(x_test,dtype=mstype.float32)
    print(x_train_whole.shape)
    global data_name
    data_name=params.data_name
    run_index=params.run_idx
    # Start Train
    # eval
    model = SGAE(x_train_whole.shape[1], params.hidden_dim)
    model.set_train(False)
    param_dict = load_checkpoint(f"./saved_model/{params.data_name}_best_auc_pr_runs{run_index}.ckpt") 
    load_param_into_net(model, param_dict)
    scores, _, _ = model(x_test_whole)
    scores = scores.asnumpy()
    auc = roc_auc_score(y_test, scores)
    ap = average_precision_score(y_test, scores)

    print(f'val finished, AUC={auc:.3f}, AP={ap:.3f}')
    return {'AUC': f'{auc:.3f}', 'AP': f'{ap:.3f}'}

def eval_image(params):
    
    context.set_context(mode=context.PYNATIVE_MODE,pynative_synchronize=True)
    context.set_context(device_target="CPU")
    # Load data
    x_train, x_test, y_train, y_test = LoadImageData(params)
    x_train_whole=Tensor(x_train,dtype=mstype.float32)
    x_test_whole=Tensor(x_test,dtype=mstype.float32)
    print(x_train_whole.shape)
   
    
    global data_name
    data_name=params.data_name
    run_index=params.run_idx
    # Start Train
    # eval
    model = SGAE(x_train_whole.shape[1], params.hidden_dim)
    model.set_train(False)
    param_dict = load_checkpoint(f"./saved_model/{params.data_name}_best_auc_pr_runs{run_index}.ckpt") 
    load_param_into_net(model, param_dict)
    scores, _, _ = model(x_test_whole)
    scores = scores.asnumpy()
    auc = roc_auc_score(y_test, scores)
    ap = average_precision_score(y_test, scores)
    print(f'eval finished, AUC={auc:.3f}, AP={ap:.3f}')
    return {'AUC': f'{auc:.3f}', 'AP': f'{ap:.3f}'}

def eval_document(params):
    
    #global params_
    #params_=params
    
    context.set_context(mode=context.PYNATIVE_MODE,pynative_synchronize=True)
    context.set_context(device_target="CPU")
    # Load data
    dataloader = LoadDocumentData(params)
    # Experiment settings
    auc = np.zeros((dataloader.class_num,))
    ap = np.zeros((dataloader.class_num,))
    
    #global data_name
    data_name=params.data_name
    run_idx=params.run_idx

    for normal_idx in range(dataloader.class_num):


        x_train, x_test, y_train, y_test = dataloader.preprocess(normal_idx)
        x_train_whole=Tensor(x_train,dtype=mstype.float32)
        x_test_whole=Tensor(x_test,dtype=mstype.float32)
        #print(x_train_whole.shape)

        model = SGAE(x_train_whole.shape[1], params.hidden_dim)
        #model.set_train(False)

        if params.verbose and normal_idx == 0 and run_idx == 0:
            print(model)

        param_dict = load_checkpoint(f"./saved_model/{params.data_name}_best_auc_pr_runs{run_idx}_normal{normal_idx}.ckpt") 
        load_param_into_net(model, param_dict)
        scores, _, _ = model(x_test_whole)
        scores = scores.asnumpy()
        auc[normal_idx] = roc_auc_score(y_test, scores)
        ap[normal_idx] = average_precision_score(y_test, scores)

        print(f'this idx finished, AUC={np.mean(auc[normal_idx]):.3f}, AP={np.mean(ap[normal_idx]):.3f}')
        
    return {'AUC': f'{np.mean(auc):.3f}', 'AP': f'{np.mean(ap):.3f}'}

    
if __name__ == '__main__':
    
    start_time = time.time()
    time_name = str(time.strftime("%m%d")) + '_' + str(time.time()).split(".")[1][-3:]
    print(f'Time name is {time_name}')
    print(os.getcwd())
    # Total metrics
    metrics = pd.DataFrame()
    # Conduct one experiements
    #args = parameter()
    args=cfg
    #print(f'Device is {args.device.type}-{args.cuda}')
    if args.data_name in ['attack', 'bcsc', 'creditcard', 'diabetic', 'donor', 'intrusion', 'market']:
        an_metrics_dict = eval_tabular(args)
    elif args.data_name in ['reuters', '20news']:
        an_metrics_dict = eval_document(args)
    elif args.data_name in ['mnist']:
        an_metrics_dict = eval_image(args)

    metrics = pd.DataFrame(an_metrics_dict, index=[0])
    metrics.to_csv(f'{args.out_dir}{args.model_name}_{args.data_name}_{time_name}.csv')
    
    print(f'Finished!\nTotal time is {time.time()-start_time:.2f}s')
    print(f'Current time is {time.strftime("%m%d_%H%M")}')
    print(f'Results:')
    print(metrics.sort_values('AUC', ascending=False))