import numpy as np
import pandas as pd
import torch
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import pickle as pkl
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from c_lib.metrics import mask_evaluation_np

class Scaler_NYC:
    def __init__(self, train):
        """ NYC Max-Min

        Arguments:
            train {np.ndarray} -- shape(T, D, W, H)
        """
        train_temp = np.transpose(train,(0,2,1)).reshape((-1,train.shape[1]))
        self.max = np.max(train_temp,axis=0)
        self.min = np.min(train_temp,axis=0)
    def transform(self, data):
        """norm train，valid，test

        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)

        Returns:
            {np.ndarray} -- shape(T, D, W, H)
        """
        T,D,W = data.shape
        data = np.transpose(data,(0,2,1)).reshape((-1,D))
        data[:,0] = (data[:,0] - self.min[0]) / (self.max[0] - self.min[0])
        data[:,33:40] = (data[:,33:40] - self.min[33:40]) / (self.max[33:40] - self.min[33:40])
        data[:,40] = (data[:,40] - self.min[40]) / (self.max[40] - self.min[40])
        data[:,46] = (data[:,46] - self.min[46]) / (self.max[46] - self.min[46])
        data[:,47] = (data[:,47] - self.min[47]) / (self.max[47] - self.min[47])
        return np.transpose(data.reshape((T,W,-1)),(0,2,1))

    def inverse_transform(self,data):
        """
        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)

        Returns:
            {np.ndarray} --  shape (T, D, W, H)
        """
        return data*(self.max[0]-self.min[0])+self.min[0]


class Scaler_Chi:
    def __init__(self, train):
        """Chicago Max-Min

        Arguments:
            train {np.ndarray} -- shape(T, D, W, H)
        """
        train_temp = np.transpose(train,(0,2,1)).reshape((-1,train.shape[1]))
        self.max = np.max(train_temp,axis=0)
        self.min = np.min(train_temp,axis=0)
    def transform(self, data):
        """norm train，valid，test

        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)

        Returns:
            {np.ndarray} -- shape(T, D, W, H)
        """
        T,D,W = data.shape
        data = np.transpose(data,(0,2,1)).reshape((-1,D))#(T*W*H,D)
        data[:,0] = (data[:,0] - self.min[0]) / (self.max[0] - self.min[0])
        data[:,33] = (data[:,33] - self.min[33]) / (self.max[33] - self.min[33])
        data[:,39] = (data[:,39] - self.min[39]) / (self.max[39] - self.min[39])
        data[:,40] = (data[:,40] - self.min[40]) / (self.max[40] - self.min[40])
        return np.transpose(data.reshape((T,W,-1)),(0,2,1))

    def inverse_transform(self,data):
        """
        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)

        Returns:
            {np.ndarray} --  shape(T, D, W, H)
        """
        return data*(self.max[0]-self.min[0])+self.min[0]


def mask_loss(predicts,labels,region_mask,data_type="nyc"):
    """

    Arguments:
        predicts {Tensor} -- predict，(batch_size,pre_len,W,H)
        labels {Tensor} -- label，(batch_size,pre_len,W,H)
        region_mask {np.array} -- mask matrix，(W,H)
        data_type {str} -- nyc/chicago

    Returns:
        {Tensor} -- MSELoss,(1,)
    """
    _,_,w,h=predicts.shape
    if w==20:
        batch_size,pre_len,w,h = predicts.shape
        predicts=predicts.reshape(batch_size,w*h)
        predicts=torch.matmul(predicts,trans1)

    else:
        batch_size,pre_len,w,h = predicts.shape
        predicts=predicts.reshape(batch_size,w*h)
        predicts=torch.matmul(predicts,trans2)


    region_mask = torch.from_numpy(region_mask).to(predicts.device)
    region_mask /= region_mask.mean()
    predicts=predicts.reshape(batch_size,1,-1)
    loss = ((labels-predicts))**2

    return (torch.mean(loss))


def mask_loss2(predicts,labels,region_mask):

    region_mask=np.dot(region_mask,trans3.transpose(1,0))
    region_mask = torch.from_numpy(region_mask).to(predicts.device)
    region_mask /= region_mask.mean()
    region_mask=region_mask.reshape(10,10)
    region_mask=region_mask
    loss = ((labels-predicts))**2
    return torch.mean(loss)





@torch.no_grad()
def compute_loss(net,f_dataloder,c_dataloder,f_risk_mask,c_risk_mask,f_road_adj,c_road_adj,f_risk_adj,c_risk_adj,f_poi_adj,c_poi_adj,grid_node_map_f,grid_node_map_c,global_step,device,trans,data_type='nyc'):
    """compute val/test loss

    Arguments:
        net {Molde} -- model
        dataloader {DataLoader} -- val/test dataloader
        risk_mask {np.array} -- mask matrix，shape(W,H)
        road_adj  {np.array} -- road adjacent matrix，shape(N,N)
        risk_adj  {np.array} -- risk adjacent matrix，shape(N,N)
        poi_adj  {np.array} -- poi adjacent matrix，shape(N,N)
        global_step {int} -- global_step
        device {Device} -- GPU

    Returns:
        np.float32 -- mean loss
    """
    net.eval()
    temp_f = []
    temp_c=[]
    for (c_feature,c_target_time,c_gragh_feature,c_label),(f_feature,f_target_time,f_gragh_feature,f_label) in zip(c_dataloder,f_dataloder):
        c_feature,c_target_time,c_graph_feature,c_label = c_feature.to(device),c_target_time.to(device),c_gragh_feature.to(device),c_label.to(device)    
        f_feature,f_target_time,f_graph_feature,f_label = f_feature.to(device),f_target_time.to(device),f_gragh_feature.to(device),f_label.to(device)
        output_f,output_c=net(f_feature,c_feature,f_target_time,f_graph_feature,c_graph_feature,f_road_adj,c_road_adj,f_risk_adj,c_risk_adj,f_poi_adj,c_poi_adj,grid_node_map_f,grid_node_map_c,trans)
        f_l = mask_loss(output_f,f_label,f_risk_mask,data_type=data_type)
        c_l = mask_loss(output_c,c_label,c_risk_mask,data_type=data_type)
        temp_f.append(f_l.cpu().item())
        loss_mean_f = sum(temp_f) / len(temp_f)
        temp_c.append(c_l.cpu().item())
        loss_mean_c = sum(temp_c) / len(temp_c)
    return loss_mean_f,loss_mean_c




@torch.no_grad()
def predict_and_evaluate(net,f_test_loader,c_test_loader,f_risk_mask,c_risk_mask,f_road_adj,c_road_adj,f_risk_adj,c_risk_adj,f_poi_adj,c_poi_adj,grid_node_map_f,grid_node_map_c,global_step,f_scaler,c_scaler,trans,device):

    net.eval()
    c_prediction_list = []
    f_prediction_list=[]
    c_label_list = []
    f_label_list = []

    for (c_feature,c_target_time,c_gragh_feature,c_label),(f_feature,f_target_time,f_gragh_feature,f_label) in zip(c_test_loader,f_test_loader):
        c_feature,c_target_time,c_graph_feature,c_label = c_feature.to(device),c_target_time.to(device),c_gragh_feature.to(device),c_label.to(device)

        f_feature,f_target_time,f_graph_feature,f_label = f_feature.to(device),f_target_time.to(device),f_gragh_feature.to(device),f_label.to(device)

        f_prediction,c_prediction=net(f_feature,c_feature,f_target_time,f_graph_feature,c_graph_feature,f_road_adj,c_road_adj,f_risk_adj,c_risk_adj,f_poi_adj,c_poi_adj,grid_node_map_f,grid_node_map_c,trans)

        c_prediction_list.append(c_prediction.cpu().numpy())
        f_prediction_list.append(f_prediction.cpu().numpy())
        c_label_list.append(c_label.cpu().numpy())
        f_label_list.append(f_label.cpu().numpy())


    f_prediction = np.concatenate(f_prediction_list, 0)
    f_label = np.concatenate(f_label_list, 0)
    f_inverse_trans_pre = f_scaler.inverse_transform(f_prediction)
    f_inverse_trans_label = f_scaler.inverse_transform(f_label)
    c_prediction = np.concatenate(c_prediction_list, 0)
    c_label = np.concatenate(c_label_list, 0)
    c_inverse_trans_pre = c_scaler.inverse_transform(c_prediction)
    c_inverse_trans_label = c_scaler.inverse_transform(c_label)

    c_rmse_,c_recall,c_recall2,c_map,c_mae= mask_evaluation_np(c_inverse_trans_label,c_inverse_trans_pre,c_risk_mask,0)
    f_rmse_,f_recall,f_recall2,f_map,f_mae= mask_evaluation_np(f_inverse_trans_label,f_inverse_trans_pre,f_risk_mask,0)
    return f_rmse_,f_recall2,f_map,f_inverse_trans_pre,f_inverse_trans_label,c_rmse_,c_recall2,c_map,c_inverse_trans_pre,c_inverse_trans_label



class Scaler_NYC2:
    def __init__(self, train):

        train_temp = np.transpose(train,(0,2,3,1)).reshape((-1,train.shape[1]))
        self.max = np.max(train_temp,axis=0)
        self.min = np.min(train_temp,axis=0)
    def transform(self, data):
  
        T,D,W,H = data.shape
        data = np.transpose(data,(0,2,3,1)).reshape((-1,D))
        data[:,0] = (data[:,0] - self.min[0]) / (self.max[0] - self.min[0])
        data[:,33:40] = (data[:,33:40] - self.min[33:40]) / (self.max[33:40] - self.min[33:40])
        data[:,40] = (data[:,40] - self.min[40]) / (self.max[40] - self.min[40])
        data[:,46] = (data[:,46] - self.min[46]) / (self.max[46] - self.min[46])
        data[:,47] = (data[:,47] - self.min[47]) / (self.max[47] - self.min[47])
        return np.transpose(data.reshape((T,W,H,-1)),(0,3,1,2))

    def inverse_transform(self,data):
        """
        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)

        Returns:
            {np.ndarray} --  shape (T, D, W, H)
        """
        return data*(self.max[0]-self.min[0])+self.min[0]


class Scaler_Chi2:
    def __init__(self, train):
        """Chicago Max-Min

        Arguments:
            train {np.ndarray} -- shape(T, D, W, H)
        """
        train_temp = np.transpose(train,(0,2,3,1)).reshape((-1,train.shape[1]))
        self.max = np.max(train_temp,axis=0)
        self.min = np.min(train_temp,axis=0)
    def transform(self, data):
        """norm train，valid，test

        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)

        Returns:
            {np.ndarray} -- shape(T, D, W, H)
        """
        T,D,W,H = data.shape
        data = np.transpose(data,(0,2,3,1)).reshape((-1,D))#(T*W*H,D)
        data[:,0] = (data[:,0] - self.min[0]) / (self.max[0] - self.min[0])
        data[:,33] = (data[:,33] - self.min[33]) / (self.max[33] - self.min[33])
        data[:,39] = (data[:,39] - self.min[39]) / (self.max[39] - self.min[39])
        data[:,40] = (data[:,40] - self.min[40]) / (self.max[40] - self.min[40])
        return np.transpose(data.reshape((T,W,H,-1)),(0,3,1,2))

    def inverse_transform(self,data):
        """
        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)

        Returns:
            {np.ndarray} --  shape(T, D, W, H)
        """
        return data*(self.max[0]-self.min[0])+self.min[0]
