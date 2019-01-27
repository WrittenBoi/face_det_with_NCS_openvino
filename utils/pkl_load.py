import pickle
import os,sys
import numpy as np


with open("val_pic.pkl", "rb") as f:
    val_pic = pickle.load(f)

with open("real_pic.pkl", "rb") as f:
    real_pic = pickle.load(f)

with open("fake_pic.pkl", "rb") as f:
    fake_pic = pickle.load(f)

def o_dis(a, b):
    dist = np.linalg.norm(a - b)  
    #dist = 1.0 / (1.0 + dist) 
    return dist

def o_dis_2(a, b):
    return np.dot(a, b)    

def cal_con_dis(a, b):
    ret = np.dot(a,b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    cos = ret/(na*nb)
    return cos

print("O dis:")
print("real", o_dis(val_pic,real_pic))
print("fake", o_dis(val_pic,fake_pic))
print("real", o_dis(real_pic,val_pic))
print("fake", o_dis(fake_pic,val_pic))
print("Cos dis:")
print("real", cal_con_dis(val_pic,real_pic))
print("fake", cal_con_dis(val_pic,fake_pic))
print("real", cal_con_dis(real_pic, val_pic))
print("fake", cal_con_dis(fake_pic, val_pic))
