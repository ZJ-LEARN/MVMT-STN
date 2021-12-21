import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import json
from torch.autograd import Variable
import math
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

###############################################
# Forked from https://github.com/Echohhhhhh/GSNet
###############################################

# (poi,road,risk)
def attention(q, k, v, d_k, dropout=None):
    #  mask(head ,seq_len ,seq_len) mask = torch.triu(torch.ones(head,seq_len, seq_len), diagonal=0)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    # print(scores.shape)
    scores = F.softmax(scores, dim=-1)
    output = torch.matmul(scores, v)

    return output

class Muview_Attention(nn.Module):
    # multiheadattention (batchsize,graph,node_num,hidden_size)
    def __init__(self,GCN_size,d_model):
        super().__init__()

        self.d_model = d_model
        self.q_linear = nn.Linear(GCN_size, d_model)
        self.k_linear = nn.Linear(GCN_size, d_model)
        self.S_linear=  nn.Linear(d_model, d_model)
        self.a1=torch.nn.Parameter(torch.Tensor(1))
        self.a2=torch.nn.Parameter(torch.Tensor(1))
        self.a3=torch.nn.Parameter(torch.Tensor(1))


    def forward(self, risk_in,road_in,poi_in):
        output=[]
        bs,num,D=road_in.shape
        v=torch.cat([risk_in,road_in,poi_in],dim=0).reshape(3,-1,num,D)
        for i in range(3):
            k = self.k_linear(v)
            q = self.q_linear(v)
            outs= attention(q, k, v, self.d_model)
            output.append(outs)

        risk_out=((output[0])[0,:,:,:]+(output[1])[0,:,:,:]+(output[2])[0,:,:,:])/3
        road_out=((output[0])[1,:,:,:]+(output[1])[1,:,:,:]+(output[2])[1,:,:,:])/3
        poi_out=((output[0])[2,:,:,:]+(output[1])[2,:,:,:]+(output[2])[2,:,:,:])/3

        risk=self.a1[0]*(risk_out)+(1-self.a1[0])*risk_in
        road=self.a2[0]*(road_out)+(1-self.a2[0])*road_in
        poi=self.a3[0]*(poi_out)+(1-self.a3[0])*poi_in
        last_out=self.S_linear(risk+road+poi)
        return last_out



class GCN_Layer(nn.Module):
    def __init__(self,num_of_features,num_of_filter):
        """One layer of GCN

        Arguments:
            num_of_features {int} -- the dimension of node feature
            num_of_filter {int} -- the number of graph filters
        """
        super(GCN_Layer,self).__init__()
        self.gcn_layer = nn.Sequential(
            nn.Linear(in_features = num_of_features,
                    out_features = num_of_filter),
            nn.ReLU()
            # nn.LeakyReLU()
        )
    def forward(self,input,adj):
        """GCN

        Arguments:
            input {Tensor} -- signal matrix,shape (batch_size,N,T*D)
            adj {np.array} -- adjacent matrix，shape (N,N)
        Returns:
            {Tensor} -- output,shape (batch_size,N,num_of_filter)
        """
        batch_size,_,_ = input.shape
        adj = torch.from_numpy(adj).to(input.device)
        adj = adj.repeat(batch_size,1,1)
        input = torch.bmm(adj, input)
        output = self.gcn_layer(input)
        return output



class Muti_GCN(nn.Module):
    def __init__(self,num_of_graph_feature,nums_of_graph_filters):

        super(Muti_GCN,self).__init__()
        self.road_gcn = nn.ModuleList()
        self.mymodel=Muview_Attention(64,64)

        for idx,num_of_filter in enumerate(nums_of_graph_filters):
            if idx == 0:
                self.road_gcn.append(GCN_Layer(num_of_graph_feature,num_of_filter))
            else:
                self.road_gcn.append(GCN_Layer(nums_of_graph_filters[idx-1],num_of_filter))

        self.risk_gcn = nn.ModuleList()
        for idx,num_of_filter in enumerate(nums_of_graph_filters):
            if idx == 0:
                self.risk_gcn.append(GCN_Layer(num_of_graph_feature,num_of_filter))
            else:
                self.risk_gcn.append(GCN_Layer(nums_of_graph_filters[idx-1],num_of_filter))

        self.poi_gcn = nn.ModuleList()
        for idx,num_of_filter in enumerate(nums_of_graph_filters):
            if idx == 0:
                self.poi_gcn.append(GCN_Layer(num_of_graph_feature,num_of_filter))
            else:
                self.poi_gcn.append(GCN_Layer(nums_of_graph_filters[idx-1],num_of_filter))
        self.mymodel=Muview_Attention(64,64)


    def forward(self,graph_feature,road_adj,risk_adj,poi_adj):

        batch_size,T,D1,N = graph_feature.shape

        road_graph_output = graph_feature.view(-1,D1,N).permute(0,2,1).contiguous()
        for gcn_layer in self.road_gcn:
            road_graph_output = gcn_layer(road_graph_output,road_adj)


        risk_graph_output = graph_feature.view(-1,D1,N).permute(0,2,1).contiguous()
        for gcn_layer in self.risk_gcn:
            risk_graph_output = gcn_layer(risk_graph_output,risk_adj)


        if poi_adj is not None:
            poi_graph_output = graph_feature.view(-1,D1,N).permute(0,2,1).contiguous()
            for gcn_layer in self.poi_gcn:
                poi_graph_output = gcn_layer(poi_graph_output,poi_adj)

        graph_output=self.mymodel(risk_graph_output,road_graph_output,poi_graph_output)
  
        graph_output=graph_output.reshape(batch_size,T,-1,N)


        return graph_output




class Time_pro(nn.Module):
    def __init__(self,
                seq_len,num_of_lstm_layers,lstm_hidden_size,
                num_of_target_time_feature):

        super(Time_pro,self).__init__()

        self.graph_lstm = nn.LSTM(64,lstm_hidden_size,num_of_lstm_layers,batch_first=True)
        self.graph_att_fc1 = nn.Linear(in_features=lstm_hidden_size,out_features=1)
        self.graph_att_fc2 = nn.Linear(in_features=num_of_target_time_feature,out_features=seq_len)
        self.graph_att_bias = nn.Parameter(torch.zeros(1))
        self.graph_att_softmax = nn.Softmax(dim=-1)
        self.lstm_hidden_size = lstm_hidden_size
        self.q_linear = nn.Linear(lstm_hidden_size, lstm_hidden_size)
        self.k_linear = nn.Linear(lstm_hidden_size, lstm_hidden_size)

    def forward(self,graph_output,
                target_time_feature):


        batch_size,T,D1,N = graph_output.shape
        graph_output = graph_output.view(batch_size,T,N,-1)\
                                    .permute(0,2,1,3)\
                                    .contiguous()\
                                    .view(batch_size*N,T,-1)
        graph_output,_ = self.graph_lstm(graph_output)
        k = self.k_linear(graph_output)
        q = self.q_linear(graph_output)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.lstm_hidden_size)
        scores = F.softmax(scores, dim=-1)
        output = torch.matmul(scores, graph_output)

        graph_target_time = torch.unsqueeze(target_time_feature,1).repeat(1,N,1).view(batch_size*N,-1)
        graph_att_fc1_output = torch.squeeze(self.graph_att_fc1(output))
        graph_att_fc2_output = self.graph_att_fc2(graph_target_time)

        graph_att_score = self.graph_att_softmax(F.relu(graph_att_fc1_output+graph_att_fc2_output+self.graph_att_bias))

        graph_att_score = graph_att_score.view(batch_size*N,-1,1)

        graph_output = torch.sum(output * graph_att_score,dim=1)
        time_output = graph_output.view(batch_size,-1,N).contiguous()


        return time_output



class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEBlock(nn.Module):
    def __init__(self, in_features):
        super(SEBlock, self).__init__()

        conv_block = [nn.Conv2d(in_features, in_features, 3, 1, 1),
                      nn.BatchNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.Conv2d(in_features, in_features, 3, 1, 1),
                      nn.BatchNorm2d(in_features),
                      nn.ReLU(),
                      ]

        self.se = SELayer(in_features)
        self.conv_block = nn.Sequential(*conv_block)


    def forward(self, x):
        out = self.conv_block(x)
        out = self.se(out)
        return x + out





class STGeoModule2(nn.Module):
    def __init__(self,grid_in_channel,num_of_lstm_layers,seq_len,
                lstm_hidden_size,num_of_target_time_feature):
        """[summary]

        Arguments:
            grid_in_channel {int} -- the number of grid data feature (batch_size,T,D,W,H),grid_in_channel=D
            num_of_lstm_layers {int} -- the number of LSTM layers
            seq_len {int} -- the time length of input
            lstm_hidden_size {int} -- the hidden size of LSTM
            num_of_target_time_feature {int} -- the number of target time feature，为24(hour)+7(week)+1(holiday)=32
        """
        super(STGeoModule2,self).__init__()
        self.grid_lstm = nn.LSTM(grid_in_channel,lstm_hidden_size,num_of_lstm_layers,batch_first=True)
        self.grid_att_fc1 = nn.Linear(in_features=lstm_hidden_size,out_features=1)
        self.grid_att_fc2 = nn.Linear(in_features=num_of_target_time_feature,out_features=seq_len)
        self.grid_att_bias = nn.Parameter(torch.zeros(1))
        self.grid_att_softmax = nn.Softmax(dim=-1)
        # self.grid1 = nn.Conv2d(in_channels=48,out_channels=64,kernel_size=3,padding=1)
        # self.grid2 = nn.Conv2d(in_channels=64,out_channels=48,kernel_size=3,padding=1)
        self.SE=SEBlock(grid_in_channel)


    def forward(self,grid_input,target_time_feature):
        batch_size,T,D,W,H = grid_input.shape
        grid_input = grid_input.reshape(-1,D,H,W)
        conv_output=self.SE(grid_input)

        conv_output = conv_output.view(batch_size,-1,D,W,H)\
                        .permute(0,3,4,1,2)\
                        .contiguous()\
                        .view(-1,T,D)
        lstm_output,_ = self.grid_lstm(conv_output)

        grid_target_time = torch.unsqueeze(target_time_feature,1).repeat(1,W*H,1).view(batch_size*W*H,-1)
        grid_att_fc1_output = torch.squeeze(self.grid_att_fc1(lstm_output))
        grid_att_fc2_output = self.grid_att_fc2(grid_target_time)
        grid_att_score = self.grid_att_softmax(F.relu(grid_att_fc1_output+grid_att_fc2_output+self.grid_att_bias))
        grid_att_score = grid_att_score.view(batch_size*W*H,-1,1)
        grid_output = torch.sum(lstm_output * grid_att_score,dim=1)
        grid_output = grid_output.view(batch_size,-1,W,H).contiguous()
        return grid_output



class MVMT(nn.Module):
    def __init__(self,grid_in_channel,num_of_lstm_layers,seq_len,pre_len,
                lstm_hidden_size,num_of_target_time_feature,
                num_of_graph_feature,nums_of_graph_filters,
                north_south_map,west_east_map):

        super(MVMT,self).__init__()
        self.st_geo_module = STGeoModule2(grid_in_channel,3,seq_len,
                                          64,num_of_target_time_feature)
        self.st_geo_module2 = STGeoModule2(grid_in_channel,num_of_lstm_layers,seq_len,
                                        lstm_hidden_size,num_of_target_time_feature)
        self.Muti_GCN_f=Muti_GCN(num_of_graph_feature,nums_of_graph_filters)
        self.Muti_GCN_c=Muti_GCN(num_of_graph_feature,nums_of_graph_filters)
        self.time_output_f=Time_pro(seq_len,num_of_lstm_layers,lstm_hidden_size,num_of_target_time_feature)
        self.time_output_c=Time_pro(seq_len,3,64,num_of_target_time_feature)
        self.north_south_map=north_south_map
        self.west_east_map=west_east_map

        fusion_channel = 16
        self.grid_weigth_f = nn.Conv2d(in_channels=lstm_hidden_size,out_channels=fusion_channel,kernel_size=1)
        self.grid_weigth_c = nn.Conv2d(in_channels=64,out_channels=fusion_channel,kernel_size=1)
        self.graph_weigth_f = nn.Conv2d(in_channels=lstm_hidden_size,out_channels=fusion_channel,kernel_size=1)
        self.graph_weigth_c = nn.Conv2d(in_channels=64,out_channels=fusion_channel,kernel_size=1)
        # nyc
        f_num=243
        c_num=75
        batch_size=32
        self.output_layer1_f = nn.Linear(fusion_channel*north_south_map*west_east_map,pre_len*north_south_map*west_east_map)
        self.output_layer1_c = nn.Linear(fusion_channel*int((north_south_map/2)*(west_east_map/2)),pre_len*int((north_south_map/2)*(west_east_map/2)))


    def forward(self,f_train_feature,c_train_feature,target_time_feature,f_graph_feature,c_graph_feature,
                f_road_adj,c_road_adj,f_risk_adj,c_risk_adj,f_poi_adj,c_poi_adj,grid_node_map_f,grid_node_map_c,trans):

# grid_output
        batch_size,_,_,_,_=c_train_feature.shape
        f_grid_output = self.st_geo_module2(f_train_feature,target_time_feature)
# grid_output
        c_grid_output = self.st_geo_module(c_train_feature,target_time_feature)


# graph_output:
        c_graph_output=self.Muti_GCN_c(c_graph_feature,c_road_adj,c_risk_adj,c_poi_adj)
        f_graph_output=self.Muti_GCN_f(f_graph_feature,f_road_adj,f_risk_adj,f_poi_adj)


# # # coarse to finer
        batch_size1,T,_,c_N=c_graph_output.shape
        batch_size,T,_,f_N=f_graph_output.shape
        c_graph_output=c_graph_output.reshape(batch_size1*T,-1,c_N)
        cf_out=F.relu(torch.matmul(c_graph_output,trans))
        f1_graph_output=f_graph_output+0.2*cf_out.reshape(batch_size1,T,-1,f_N)

# finer to coarse
        f_graph_output=f_graph_output.reshape(batch_size*T,-1,f_N)
        trans2=trans.permute(0,2,1)
        fc_out=F.relu(torch.matmul(f_graph_output,trans2))
        c_graph_output=c_graph_output.reshape(batch_size1,T,-1,c_N)
        c1_graph_output=c_graph_output+0.8*fc_out.reshape((batch_size,T,-1,c_N))
        graph_output_c1=self.time_output_c(c1_graph_output,target_time_feature)
        graph_output_f1=self.time_output_f(f1_graph_output,target_time_feature)
        graph_output_c=graph_output_c1
        graph_output_f=graph_output_f1
        graph_output_f=graph_output_f.permute(0,2,1)
        batch_size,_,_=graph_output_f.shape
        grid_node_map_tmp_f = torch.from_numpy(grid_node_map_f)\
                            .to(graph_output_f.device)\
                            .repeat(batch_size,1,1)
        graph_output_f = torch.bmm(grid_node_map_tmp_f,graph_output_f)\
                            .permute(0,2,1)\
                            .view(batch_size,-1,self.north_south_map,self.west_east_map)
        graph_output_c=graph_output_c.permute(0,2,1)
        batch_size,_,_=graph_output_c.shape

        grid_node_map_tmp_c = torch.from_numpy(grid_node_map_c)\
                            .to(graph_output_f.device)\
                            .repeat(batch_size,1,1)

        graph_output_c = torch.bmm(grid_node_map_tmp_c,graph_output_c)\
                            .permute(0,2,1)\
                            .view(batch_size,-1,int((self.north_south_map)/2),int((self.west_east_map)/2))



        f_grid_output = self.grid_weigth_f(f_grid_output)
        graph_output_f = self.graph_weigth_f(graph_output_f)
        f_fusion_output = (f_grid_output + graph_output_f).view(batch_size,-1)
        f_final_output = self.output_layer1_f(f_fusion_output).view(batch_size,-1,self.north_south_map,self.west_east_map)
        c_grid_output = self.grid_weigth_c(c_grid_output)
        graph_output_c = self.graph_weigth_c(graph_output_c)
        c_fusion_output = (c_grid_output + graph_output_c).view(batch_size,-1)
        c_final_output = self.output_layer1_c(c_fusion_output).view(batch_size,-1,int(self.north_south_map/2),int(self.west_east_map/2))

        return f_final_output,c_final_output
























