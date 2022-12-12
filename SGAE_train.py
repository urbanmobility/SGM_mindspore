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

class CustomWithEvalCell(nn.Cell):
    """自定义多标签评估网络"""

    def __init__(self, network):
        super(CustomWithEvalCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, data, label):
        scores,x_dec,_ = self.network(data)
        recon_error = ms.numpy.mean(ms.numpy.multiply(data - x_dec,data - x_dec))
        print("recon_error",recon_error)
        return scores, label
    
class AUC_PR(nn.Metric):
    """定义metric"""

    def __init__(self):
        super(AUC_PR, self).__init__()
        self.clear()

    def clear(self):
        """初始化变量abs_error_sum和samples_num"""

    def update(self, *inputs):
        """更新abs_error_sum和samples_num"""
        scores= inputs[0].asnumpy()
        #print("scores",scores)
        y = inputs[1].asnumpy()
        #print("y",y)

        # 计算预测值与真实值的auc和pr
        self.auc = roc_auc_score(y, scores)
        self.pr = average_precision_score(y, scores)

    def eval(self):
        """计算最终评估结果"""
        return self.auc,self.pr

class EvalCallback(Callback):
    """
    Evaluation per epoch, and save the best AUC_PR checkpoint.
    """
    def __init__(self, model, eval_ds, save_path="./"):
        
        global best_auc
        global best_pr
        
        self.model = model
        self.eval_ds = eval_ds
        self.best_auc = best_auc
        self.best_pr = best_pr
        self.save_path = save_path
        #print("init best_pr",self.best_pr)

    def epoch_end(self, run_context):
        """
        evaluate at epoch end.
        """
        global best_auc
        global best_pr
        
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        #res = self.model.eval(self.eval_ds, dataset_sink_mode=False)
        res = self.model.eval(self.eval_ds)
        auc = res["auc_pr"][0]
        pr = res["auc_pr"][1]
        if auc+pr > self.best_auc + self.best_pr:
        #if auc > self.best_auc:
            self.best_pr = pr
            self.best_auc = auc
            best_auc=auc
            best_pr=pr
            if params_.data_name not in ['reuters', '20news']:
                ms.save_checkpoint(cb_params.train_network, os.path.join(self.save_path, f"{data_name}_best_auc_pr_runs{run_index}.ckpt"))
            else:
                ms.save_checkpoint(cb_params.train_network, os.path.join(self.save_path, f"{data_name}_best_auc_pr_runs{run_index}_normal{normal_index}.ckpt"))
                
            print("the best epoch is", cur_epoch, "best auc pr is", self.best_auc,self.best_pr)

class CustomWithLossCell(nn.Cell):
    def __init__(self, backbone,norm_thresh,params):
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self.norm_thresh=norm_thresh
        self.params=params

    def construct(self, x, label):
        
        scores, x_dec, _= self._backbone(x)
        anomal_flag = recog_anomal(x, x_dec, self.norm_thresh)
        #recon_error = ms.numpy.mean((x - x_dec) ** 2)
        recon_error = ms.numpy.mean(ms.numpy.multiply(x - x_dec,x - x_dec))
        #print("recon_error",recon_error)
        dist_error = self.compute_dist_error(scores, Tensor(anomal_flag,mstype.int32) ,self.params)
        #print("dist_error",dist_error)
        loss = recon_error + self.params.lam_dist * dist_error
        #print("loss",loss)
        return loss
    
    def compute_dist_error(self, scores, anomal_flag, params):
        # inlier loss
        ref=ms.numpy.randn((1000,))
        dev = scores - ms.numpy.mean(ref)
        inlier_loss = ms.numpy.absolute(dev)
        # outlier loss
        anomal_flag = ms.numpy.expand_dims(anomal_flag,1)
        outlier_loss = ms.numpy.absolute(ms.numpy.maximum(params.a - scores, ms.numpy.zeros(scores.shape)))
        dist_error = ms.numpy.mean((1 - anomal_flag) * inlier_loss + params.lam_out * anomal_flag * outlier_loss)
        return dist_error

def train_tabular(params):

    global params_
    params_=params
    
    context.set_context(mode=context.PYNATIVE_MODE,pynative_synchronize=True)
    context.set_context(device_target="CPU")
    # Load data

    
    for seed_idx in range(10):
        params.np_seed+=1
        print("*"*30+"np_seed",params.np_seed)

        x_train, y_train, x_val, y_val, x_test, y_test = LoadTabularData(params)
        x_train_whole=Tensor(x_train,dtype=mstype.float32)
        x_test_whole=Tensor(x_test,dtype=mstype.float32)
        print(x_train_whole.shape)
        auc = np.zeros(params.run_num)
        ap = np.zeros(params.run_num)
        global data_name
        data_name=params.data_name

        
        # Start Train
        for run_idx in tqdm(range(params.run_num)):
            start_time = time.time()
            global best_auc
            global best_pr
            global run_index
            run_index=run_idx
            best_auc = 0
            best_pr = 0
            model = SGAE(x_train_whole.shape[1], params.hidden_dim)
            optim = nn.Adam(model.trainable_params(), learning_rate=params.lr)

            if params.verbose and run_idx == 0:
                print(model)

            # One run
            for epoch in range(params.epochs): 
            #if True:
                ds_train=create_dataset(x_train,y_train,params)
                ds_val=create_dataset(x_val,y_val,params,is_batch=False)
                ds_test=create_dataset(x_test,y_test,params,is_batch=False)
                #print(ds_val.get_batch_size())
                #print(ds_train.get_dataset_size())

                epoch_time_start = time.time() 
                #print("epoch_time_start",epoch_time_start)
                # train

                # calculate norm thresh
                _, dec_train, _ = model(x_train_whole)
                norm = calculate_norm(x_train_whole, dec_train)
                norm_thresh = np.percentile(norm, params.epsilon)
                #print("norm_thresh",norm_thresh)

                loss = 0
                recon_error = 0
                dist_error = 0

                auc_pr=AUC_PR()
                #auc_pr.set_indexes([0,3])
                model_withloss=CustomWithLossCell(model,norm_thresh,params)
                eval_net = CustomWithEvalCell(model)
                model_withloss = Model(model_withloss, optimizer=optim,eval_network=eval_net,metrics={'auc_pr':auc_pr})
                #model_withloss = Model(model_withloss, optimizer=optim)
                eval_callback = EvalCallback(model_withloss, ds_val, save_path="./saved_model")
                model_withloss.train(epoch=1, train_dataset=ds_train, callbacks=[TimeMonitor(30),LossMonitor(100),eval_callback],dataset_sink_mode=False)
                #result = model_withloss.eval(ds_val)
                #print(result)
                #model_withloss.train(epoch=1, train_dataset=ds_train, callbacks=[TimeMonitor(30),LossMonitor(30)],dataset_sink_mode=False)

                epoch_time = time.time() - epoch_time_start



                if params.verbose:
                    if (epoch + 1) % params.print_step == 0 or epoch == 0:
                        scores, _, _ = model(x_test_whole)
                        scores = scores.asnumpy()
                        auc_ = roc_auc_score(y_test, scores)
                        ap_ = average_precision_score(y_test, scores)
                        print(f'Epoch num:[{epoch+1}/{params.epochs}], Time:{epoch_time:.3f} ' +\
                                f'--Loss:{loss:.3f}, --RE:{recon_error:.3f}, --DE:{dist_error:.3f}, --DE_r:{dist_error*params.lam_dist:.3f},'+\
                                f'--AUC:{auc_:.3f} --AP:{ap_:.3f}')    

                '''
                # Early Stop
                if params.early_stop:
                    scores, _, _ = model(x_train_whole)
                    scores = scores.asnumpy()   
                    if np.mean(scores) > params.a / 2:
                        print(f'Early Stop at Epoch={epoch+1}, AUC={auc[run_idx]:.3f}')
                        break
                '''

            # test
            param_dict = load_checkpoint(f"./saved_model/{params.data_name}_best_auc_pr_runs{run_index}.ckpt") 
            load_param_into_net(model, param_dict)
            scores, _, _ = model(x_test_whole)
            scores = scores.asnumpy()
            auc[run_idx] = roc_auc_score(y_test, scores)
            ap[run_idx] = average_precision_score(y_test, scores)

            print(f'This run finished, AUC={auc[run_idx]:.3f}, AP={ap[run_idx]:.3f}')

            # RUN JUMP
            if run_idx > 5 and np.mean(auc[:run_idx]) < 0.5:
                print('RUN JUMP')
                print(f'Average AUC is : {np.mean(auc[:run_idx]):.3f}')
                print(f'AUC is : {auc}')
                break

        print(f'Train Finished, AUC={np.mean(auc):.3f}({np.std(auc):.3f}), AP={np.mean(ap):.3f}({np.std(ap):.3f}),np_seed={params.np_seed}')
        
    return {'AUC': f'{np.mean(auc):.3f}({np.std(auc):.3f})', 'AP': f'{np.mean(ap):.3f}({np.std(ap):.3f})'}

def train_image(params):
    
    global params_
    params_=params
    
    context.set_context(mode=context.PYNATIVE_MODE,pynative_synchronize=True)
    context.set_context(device_target="CPU")
    # Load data
    x_train, x_test, y_train, y_test = LoadImageData(params)
    x_train_whole=Tensor(x_train,dtype=mstype.float32)
    x_test_whole=Tensor(x_test,dtype=mstype.float32)
    print(x_train_whole.shape)
    
    # Experiment settings
    auc = np.zeros(params.run_num)
    ap = np.zeros(params.run_num)
    
    global data_name
    data_name=params.data_name
    
    # Start Train
    for run_idx in tqdm(range(params.run_num)):
        start_time = time.time()
        global best_auc
        global best_pr
        global run_index
        run_index=run_idx
        best_auc = 0
        best_pr = 0
        model = SGAE(x_train_whole.shape[1], params.hidden_dim)
        optim = nn.Adam(model.trainable_params(), learning_rate=params.lr)

        if params.verbose and run_idx == 0:
            print(model)

        # One run
        for epoch in range(params.epochs): 
            ds_train=create_dataset(x_train,y_train,params)
            #ds_val=create_dataset(x_val,y_val,params,is_batch=False)
            ds_val=create_dataset(x_test,y_test,params,is_batch=False)
            ds_test=create_dataset(x_test,y_test,params,is_batch=False)
            #print(ds_val.get_batch_size())
            #print(ds_train.get_dataset_size())
            
            epoch_time_start = time.time() 
            # train
            
            # calculate norm thresh
            _, dec_train, _ = model(x_train_whole)
            norm = calculate_norm(x_train_whole, dec_train)
            norm_thresh = np.percentile(norm, params.epsilon)
            #print("norm_thresh",norm_thresh)

            loss = 0
            recon_error = 0
            dist_error = 0
             
            auc_pr=AUC_PR()
            #auc_pr.set_indexes([0,3])
            model_withloss=CustomWithLossCell(model,norm_thresh,params)
            eval_net = CustomWithEvalCell(model)
            model_withloss = Model(model_withloss, optimizer=optim,eval_network=eval_net,metrics={'auc_pr':auc_pr})
            #model_withloss = Model(model_withloss, optimizer=optim)
            eval_callback = EvalCallback(model_withloss, ds_val, save_path="./saved_model")
            model_withloss.train(epoch=1, train_dataset=ds_train, callbacks=[TimeMonitor(30),LossMonitor(30),eval_callback],dataset_sink_mode=False)
            

            #nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
                
            epoch_time = time.time() - epoch_time_start 

            # test
            if params.verbose:
                if (epoch + 1) % params.print_step == 0 or epoch == 0:
                    scores, _, _ = model(x_test_whole)
                    scores = scores.asnumpy()
                    auc_ = roc_auc_score(y_test, scores)
                    ap_ = average_precision_score(y_test, scores)
                    print(f'Epoch num:[{epoch+1}/{params.epochs}], Time:{epoch_time:.3f} ' +\
                            f'--Loss:{loss:.3f}, --RE:{recon_error:.3f}, --DE:{dist_error:.3f}, --DE_r:{dist_error*params.lam_dist:.3f},'+\
                            f'--AUC:{auc_:.3f} --AP:{ap_:.3f}')    

            '''
            # Early Stop
            if params.early_stop:
                scores, _, _ = model(x_train_whole)
                scores = scores.asnumpy()   
                if np.mean(scores) > params.a / 2:
                    print(f'Early Stop at Epoch={epoch+1}, AUC={auc[run_idx]:.3f}')
                    break
            '''

        # test
        param_dict = load_checkpoint(f"./saved_model/{params.data_name}_best_auc_pr_runs{run_index}.ckpt") 
        load_param_into_net(model, param_dict)
        scores, _, _ = model(x_test_whole)
        scores = scores.asnumpy()
        auc[run_idx] = roc_auc_score(y_test, scores)
        ap[run_idx] = average_precision_score(y_test, scores)
         
        print(f'This run finished, AUC={auc[run_idx]:.3f}, AP={ap[run_idx]:.3f}')
      
        # RUN JUMP
        if run_idx > 5 and np.mean(auc[:run_idx]) < 0.5:
            print('RUN JUMP')
            print(f'Average AUC is : {np.mean(auc[:run_idx]):.3f}')
            print(f'AUC is : {auc}')
            break
    
    print(f'Train Finished, AUC={np.mean(auc):.3f}({np.std(auc):.3f}), AP={np.mean(ap):.3f}({np.std(ap):.3f})')
    return {'AUC': f'{np.mean(auc):.3f}({np.std(auc):.3f})', 'AP': f'{np.mean(ap):.3f}({np.std(ap):.3f})'}

def train_document(params):
    
    global params_
    params_=params
    
    context.set_context(mode=context.PYNATIVE_MODE,pynative_synchronize=True)
    context.set_context(device_target="CPU")
    # Load data
    dataloader = LoadDocumentData(params)
    
    # Experiment settings
    auc = np.zeros((params.run_num, dataloader.class_num))
    ap = np.zeros((params.run_num, dataloader.class_num))
    
    global data_name
    data_name=params.data_name
    
    # Start Train
    for run_idx in tqdm(range(params.run_num)):
        
        start_time = time.time()
        global run_index
        run_index=run_idx
        print("class_num",dataloader.class_num)
        # Iterate for normal class
        for normal_idx in range(dataloader.class_num):
            
            global best_auc
            global best_pr
            global normal_index
            best_auc = 0
            best_pr = 0
            normal_index=normal_idx
            
            x_train, x_test, y_train, y_test = dataloader.preprocess(normal_idx)
            x_train_whole=Tensor(x_train,dtype=mstype.float32)
            x_test_whole=Tensor(x_test,dtype=mstype.float32)
            #print(x_train_whole.shape)

            model = SGAE(x_train_whole.shape[1], params.hidden_dim)
            optim = nn.Adam(model.trainable_params(), learning_rate=params.lr)

            if params.verbose and normal_idx == 0 and run_idx == 0:
                print(model)


            # One run
            for epoch in range(params.epochs): 
                
                ds_train=create_dataset(x_train,y_train,params)
                #ds_val=create_dataset(x_val,y_val,params,is_batch=False)
                ds_val=create_dataset(x_test,y_test,params,is_batch=False)
                ds_test=create_dataset(x_test,y_test,params,is_batch=False)
                
                epoch_time_start = time.time() 
                 
                # calculate norm thresh
                _, dec_train, _ = model(x_train_whole)
                norm = calculate_norm(x_train_whole, dec_train)
                norm_thresh = np.percentile(norm, params.epsilon)
                #print("norm_thresh",norm_thresh)

                loss = 0
                recon_error = 0
                dist_error = 0
                
                auc_pr=AUC_PR()
                #auc_pr.set_indexes([0,3])
                model_withloss=CustomWithLossCell(model,norm_thresh,params)
                eval_net = CustomWithEvalCell(model)
                model_withloss = Model(model_withloss, optimizer=optim,eval_network=eval_net,metrics={'auc_pr':auc_pr})
                #model_withloss = Model(model_withloss, optimizer=optim)
                eval_callback = EvalCallback(model_withloss, ds_val, save_path="./saved_model")
                model_withloss.train(epoch=1, train_dataset=ds_train, callbacks=[TimeMonitor(30),LossMonitor(1),eval_callback],dataset_sink_mode=False)


                #nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)

                epoch_time = time.time() - epoch_time_start 

                # test
                if params.verbose:
                    if (epoch + 1) % params.print_step == 0 or epoch == 0:
                        scores, _, _ = model(x_test_whole)
                        scores = scores.asnumpy()
                        auc_ = roc_auc_score(y_test, scores)
                        ap_ = average_precision_score(y_test, scores)
                        print(f'Epoch num:[{epoch+1}/{params.epochs}], Time:{epoch_time:.3f} ' +\
                                f'--Loss:{loss:.3f}, --RE:{recon_error:.3f}, --DE:{dist_error:.3f}, --DE_r:{dist_error*params.lam_dist:.3f},'+\
                                f'--AUC:{auc_:.3f} --AP:{ap_:.3f}')    

                '''
                # Early Stop
                if params.early_stop:
                    scores, _, _ = model(x_train_whole)
                    scores = scores.asnumpy()   
                    if np.mean(scores) > params.a / 2:
                        print(f'Early Stop at Epoch={epoch+1}, AUC={auc[run_idx]:.3f}')
                        break
                '''
            # test
            param_dict = load_checkpoint(f"./saved_model/{params.data_name}_best_auc_pr_runs{run_index}_normal{normal_idx}.ckpt") 
            load_param_into_net(model, param_dict)
            scores, _, _ = model(x_test_whole)
            scores = scores.asnumpy()
            auc[run_idx][normal_idx] = roc_auc_score(y_test, scores)
            ap[run_idx][normal_idx] = average_precision_score(y_test, scores)
         
        print(f'This run finished, AUC={np.mean(auc[run_idx]):.3f}, AP={np.mean(ap[run_idx]):.3f}')
      
        # RUN JUMP
        if run_idx > 5 and np.mean(auc[:run_idx]) < 0.5:
            print('RUN JUMP')
            print(f'Average AUC is : {np.mean(auc[:run_idx]):.3f}')
            print(f'AUC is : {auc}')
            break
    
    print(f'Train Finished, AUC={np.mean(auc):.3f}({np.std(auc):.3f}), AP={np.mean(ap):.3f}({np.std(ap):.3f})')
    return {'AUC': f'{np.mean(auc):.3f}({np.std(auc):.3f})', 'AP': f'{np.mean(ap):.3f}({np.std(ap):.3f})'}



def recog_anomal(data, x_dec, thresh):
    ''' Recognize anomaly
    '''
    norm = calculate_norm(data, x_dec)
    anomal_flag = norm.copy()
    anomal_flag[norm < thresh] = 0
    anomal_flag[norm >= thresh] = 1
    return anomal_flag

def calculate_norm(data, x_dec):
    ''' Calculate l2 norm
    '''
    delta = (data - x_dec).asnumpy()
    norm = np.linalg.norm(delta, ord=2, axis=1)
    return norm

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
        an_metrics_dict = train_tabular(args)
    elif args.data_name in ['reuters', '20news']:
        an_metrics_dict = train_document(args)
    elif args.data_name in ['mnist']:
        an_metrics_dict = train_image(args)

    metrics = pd.DataFrame(an_metrics_dict, index=[0])
    metrics.to_csv(f'{args.out_dir}{args.model_name}_{args.data_name}_{time_name}.csv')
    
    print(f'Finished!\nTotal time is {time.time()-start_time:.2f}s')
    print(f'Current time is {time.strftime("%m%d_%H%M")}')
    print(f'Results:')
    print(metrics.sort_values('AUC', ascending=False))