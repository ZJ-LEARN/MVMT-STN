import numpy as np
import math
import pickle as pkl
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def transfer_dtype(y_true,y_pred):
    return y_true.astype('float32'),y_pred.astype('float32')

def mask_mse_np(y_true,y_pred,region_mask,null_val=None):
    _,_,w,h=y_pred.shape
    if w==20:
        N,pre_len,w,h = y_pred.shape
        y_true,y_pred = transfer_dtype(y_true,y_pred)
        y_pred=y_pred.reshape(N,400)
        y_pred=np.matmul(y_pred,trans1)
    else:
        N,pre_len,w,h = y_pred.shape
        y_true,y_pred = transfer_dtype(y_true,y_pred)
        y_pred=y_pred.reshape(N,100)
        y_pred=np.matmul(y_pred,trans2)

    y_pred=y_pred.reshape(N,1,-1)
    if null_val is not None:
        label_mask = np.where(y_true > 0,1,0).astype('float32')
        mask = region_mask * label_mask
    else:
        mask = region_mask

    rmse= np.mean(((y_true-y_pred))**2)
    mae=np.mean(np.abs((y_true-y_pred)))

    return rmse,mae


def mask_rmse_np(y_true,y_pred,region_mask,null_val=None):

    y_true,y_pred = transfer_dtype(y_true,y_pred)
    rmse,mae=mask_mse_np(y_true,y_pred,region_mask,null_val)
    return math.sqrt(rmse),mae


def nonzero_num(y_true):

    nonzero_list = []
    threshold = 0
    for i in range(len(y_true)):
        non_zero_nums = (y_true[i] > threshold).sum()
        nonzero_list.append(non_zero_nums)
    return nonzero_list

def get_top(data,accident_nums):

    data = data.reshape((data.shape[0],-1))
    topk_list = []
    for i in range(len(data)):
        risk = {}
        for j in range(len(data[i])):
            risk[j] = data[i][j]
        k = int(accident_nums[i])
        topk_list.append(list(dict(sorted(risk.items(),key=lambda x:x[1],reverse=True)[:k]).keys()))
    return topk_list




def compute(y_pred,y_true,region_mask):
    _,_,w,h=y_pred.shape
    if w==20:
        N,pre_len,w,h = y_pred.shape
        y_true,y_pred = transfer_dtype(y_true,y_pred)
        y_pred=y_pred.reshape(N,400)
        y_pred=np.matmul(y_pred,trans1)
    else:
        N,pre_len,w,h = y_pred.shape
        y_true,y_pred = transfer_dtype(y_true,y_pred)
        y_pred=y_pred.reshape(N,100)
        y_pred=np.matmul(y_pred,trans2)
    y_pred = y_pred.reshape((y_pred.shape[0],-1))
    y_true = y_true.reshape((y_true.shape[0],-1))
    k=0
    k2=0
    for i in range(len(y_true)):
        label=list(np.flatnonzero(y_true[i]>0))
        if w==20:
            pre=list(np.argsort((y_pred[i]))[193:243])
        else:
            pre=list(np.argsort((y_pred[i]))[55:75])
        cross=list(set(pre).intersection(set(label)))
        k=k+len(cross)
        k2=k2+len(label)
    return  k*100/k2


def Get(y_true,y_pred,region_mask):


    region_mask = np.where(region_mask >= 1,0,-1000)
    tmp_y_true = y_true + region_mask
    tmp_y_pred = y_pred + region_mask

    accident_grids_nums = nonzero_num(tmp_y_true)
    
    true_top_k = get_top(tmp_y_true,accident_grids_nums)
    pred_top_k = get_top(tmp_y_pred,accident_grids_nums)
    
    hit_sum = 0
    for i in range(len(true_top_k)):
        intersection = [v for v in true_top_k[i] if v in pred_top_k[i]]
        hit_sum += len(intersection)
    return hit_sum / sum(accident_grids_nums) * 100


def MAP(y_true,y_pred,region_mask):
    _,_,w,h=y_pred.shape
    if w==20:
        N,pre_len,w,h = y_pred.shape
        y_true,y_pred = transfer_dtype(y_true,y_pred)
        y_pred=y_pred.reshape(N,400)
        y_pred=np.matmul(y_pred,trans1)
    else:
        N,pre_len,w,h = y_pred.shape
        y_true,y_pred = transfer_dtype(y_true,y_pred)
        y_pred=y_pred.reshape(N,100)
        y_pred=np.matmul(y_pred,trans2)

    region_mask = np.where(region_mask >= 1,0,-1000)
    tmp_y_true = y_true + region_mask
    tmp_y_pred = y_pred + region_mask

    accident_grids_nums = nonzero_num(tmp_y_true)

    true_top_k = get_top(tmp_y_true,accident_grids_nums)
    pred_top_k = get_top(tmp_y_pred,accident_grids_nums)

    all_k_AP = []
    for sample in range(len(true_top_k)):
        all_k_AP.append(AP(list(true_top_k[sample]),list(pred_top_k[sample])))
    return sum(all_k_AP)/len(all_k_AP)

def AP(label_list,pre_list):
    hits = 0
    sum_precs = 0
    for n in range(len(pre_list)):
        if pre_list[n] in label_list:
            hits += 1
            sum_precs += hits / (n + 1.0)
    if hits > 0:
        return sum_precs / len(label_list)
    else:
        return 0

def mask_evaluation_np(y_true,y_pred,region_mask,null_val=None):
    rmse_ ,mae= mask_rmse_np(y_true,y_pred,region_mask,null_val)
    map_ = MAP(y_true,y_pred,region_mask)
    kk=compute(y_pred,y_true,region_mask)
    return rmse_,kk,map_,mae