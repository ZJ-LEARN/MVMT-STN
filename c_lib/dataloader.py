import numpy as np
import pickle as pkl
import configparser
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from c_lib.utils import Scaler_NYC,Scaler_Chi,Scaler_NYC2,Scaler_Chi2

#high frequency time
high_fre_hour = [6,7,8,15,16,17,18]

def split_and_norm_data_time2(all_data,
                        train_rate = 0.6,
                        valid_rate = 0.2,
                        recent_prior=3,
                        week_prior=4,
                        one_day_period=24,
                        days_of_week=7,
                        pre_len=1):
    num_of_time,channel,_,_= all_data.shape
    train_line, valid_line = int(num_of_time * train_rate), int(num_of_time * (train_rate+valid_rate))
    for index,(start,end) in enumerate(((0,train_line),(train_line,valid_line),(valid_line,num_of_time))):
        if index == 0:
            if channel == 48:
                scaler = Scaler_NYC2(all_data[start:end,:,:,:])
            if channel == 41:
                scaler = Scaler_Chi2(all_data[start:end,:,:,:])
        norm_data = scaler.transform(all_data[start:end,:,:,:])
        X,Y,target_time = [],[],[]
        high_X,high_Y,high_target_time = [],[],[]

        for i in range(len(norm_data)-week_prior*days_of_week*one_day_period-pre_len+1):
            t = i+week_prior*days_of_week*one_day_period
            label = norm_data[t:t+pre_len,0,:,:]
            period_list = []
            for week in range(week_prior):
                period_list.append(i+week*days_of_week*one_day_period)
            for recent in list(range(1,recent_prior+1))[::-1]:
                period_list.append(t-recent)
            feature = norm_data[period_list,:,:,:]
            X.append(feature)
            Y.append(label)
            target_time.append(norm_data[t,1:33,0,0])
            if list(norm_data[t,1:25,0,0]).index(1) in high_fre_hour:
                high_X.append(feature)
                high_Y.append(label)
                high_target_time.append(norm_data[t,1:33,0,0])
        yield np.array(X),np.array(Y),np.array(target_time),np.array(high_X),np.array(high_Y),np.array(high_target_time),scaler

def normal_and_generate_dataset_time2(
        all_data_filename,
        train_rate=0.6,
        valid_rate=0.2,
        recent_prior=3,
        week_prior=4,
        one_day_period=24,
        days_of_week=7,
        pre_len=1):
    all_data = pkl.load(open(all_data_filename,'rb')).astype(np.float32)

    for i in split_and_norm_data_time2(all_data,
                        train_rate = train_rate,
                        valid_rate = valid_rate,
                        recent_prior = recent_prior,
                        week_prior = week_prior,
                        one_day_period = one_day_period,
                        days_of_week = days_of_week,
                        pre_len = pre_len):
        yield i



def normal_and_generate_dataset(
        all_data_filename,
        train_rate=0.6,
        valid_rate=0.2,
        recent_prior=3,
        week_prior=4,
        one_day_period=24,
        days_of_week=7,
        pre_len=1):

    risk_taxi_time_data = pkl.load(open(all_data_filename,'rb')).astype(np.float32)

    for i in split_and_norm_data(risk_taxi_time_data,
                        train_rate = train_rate,
                        valid_rate = valid_rate,
                        recent_prior = recent_prior,
                        week_prior = week_prior,
                        one_day_period = one_day_period,
                        days_of_week = days_of_week,
                        pre_len = pre_len):
        yield i

def split_and_norm_data_time(all_data,
                        train_rate = 0.6,
                        valid_rate = 0.2,
                        recent_prior=3,
                        week_prior=4,
                        one_day_period=24,
                        days_of_week=7,
                        pre_len=1):
    num_of_time,channel,_= all_data.shape
    train_line, valid_line = int(num_of_time * train_rate), int(num_of_time * (train_rate+valid_rate))
    for index,(start,end) in enumerate(((0,train_line),(train_line,valid_line),(valid_line,num_of_time))):
        if index == 0:
            if channel == 48:
                scaler = Scaler_NYC(all_data[start:end,:,:])
            if channel == 41:
                scaler = Scaler_Chi(all_data[start:end,:,:])
        norm_data = scaler.transform(all_data[start:end,:,:])
        X,Y,target_time = [],[],[]
        high_X,high_Y,high_target_time = [],[],[]

        for i in range(len(norm_data)-week_prior*days_of_week*one_day_period-pre_len+1):
            t = i+week_prior*days_of_week*one_day_period
            label = norm_data[t:t+pre_len,0,:]
            period_list = []
            for week in range(week_prior):
                period_list.append(i+week*days_of_week*one_day_period)
            for recent in list(range(1,recent_prior+1))[::-1]:
                period_list.append(t-recent)
            feature = norm_data[period_list,:,:]
            X.append(feature)
            Y.append(label)
            target_time.append(norm_data[t,1:33,0])
            if list(norm_data[t,1:25,0]).index(1) in high_fre_hour:
                high_X.append(feature)
                high_Y.append(label)
                high_target_time.append(norm_data[t,1:33,0])
        yield np.array(X),np.array(Y),np.array(target_time),np.array(high_X),np.array(high_Y),np.array(high_target_time),scaler


def normal_and_generate_dataset_time(
        all_data_filename,
        train_rate=0.6,
        valid_rate=0.2,
        recent_prior=3,
        week_prior=4,
        one_day_period=24,
        days_of_week=7,
        pre_len=1):
    all_data = pkl.load(open(all_data_filename,'rb')).astype(np.float32)

    for i in split_and_norm_data_time(all_data,
                        train_rate = train_rate,
                        valid_rate = valid_rate,
                        recent_prior = recent_prior,
                        week_prior = week_prior,
                        one_day_period = one_day_period,
                        days_of_week = days_of_week,
                        pre_len = pre_len):
        yield i

def get_mask(mask_path):
   
    mask = pkl.load(open(mask_path,'rb')).astype(np.float32)
    return mask

def get_adjacent(adjacent_path):

    adjacent = pkl.load(open(adjacent_path,'rb')).astype(np.float32)
    return adjacent


def get_grid_node_map_maxtrix(grid_node_path):

    grid_node_map = pkl.load(open(grid_node_path,'rb')).astype(np.float32)
    return grid_node_map
