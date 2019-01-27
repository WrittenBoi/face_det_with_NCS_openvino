import sys,os,cv2
from time import time
import numpy as np, math
from argparse import ArgumentParser
from openvino.inference_engine import IENetwork, IEPlugin
import pickle

debug = True

model_path =    "model/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.xml"
weights_path =  "model/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.bin"
valid_pic = "test/val_pic.jpg"
real_pic =  "test/real_pic.jpg"
fake_pic =  "test/fake_pic.jpg"
cpu_ext_path =  "lib/libcpu_extension.so"
plugin_dirs = None
#device = "CPU"
device = "MYRIAD"
confidence_tolerance = 0.3

label_text_color = (255, 255, 255)
label_background_color = (125, 175, 75)
box_color = (255, 128, 0)
box_thickness = 1


def log(*args):
    if(debug):
        print(*args)

#Plugin init
log("Plugin %s init!" % device)
plugin = IEPlugin(device=device, plugin_dirs=plugin_dirs)
if "CPU" == device:
    plugin.add_cpu_extension(cpu_ext_path)

#Read IR
if "MYRIAD" == device:
    model_path = model_path.replace("FP32", "FP16")
    weights_path= weights_path.replace("FP32", "FP16")
log("Loading network file:\n\t{}\n\t{}".format(model_path, weights_path))
net = IENetwork(model=model_path, weights=weights_path)

#Get I/O blobs
input_blob = next(iter(net.inputs))
output_blob = next(iter(net.outputs))
log("Input batch size:", net.batch_size)
log("Input Shape:", net.inputs[input_blob].shape)
log("Output Shape:", net.outputs[output_blob].shape)
n,c,h,w = net.inputs[input_blob].shape

#Load model to the plugin
t0 = time()
exec_net = plugin.load(network=net)
load_time = (time() - t0) * 1000
log("Load model to the plugin, time: %fms" % load_time)
del net

def prepare_pic(pic_path):
    org_img = cv2.imread(pic_path)
    org_h,org_w,org_c = org_img.shape
    org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(org_img,(w,h))
    img = img.transpose((2,0,1)) # Change data layout from HWC to CHW
    return img,org_img
    

#Get test pic
val_img = np.ndarray(shape=(n,c,h,w))
for i in range(n):
    img,org_img = prepare_pic(valid_pic)
    val_img[i] = img
log("Valid picture has been prepared!")

real_img = np.ndarray(shape=(n,c,h,w))
for i in range(n):
    img,org_img = prepare_pic(real_pic)
    real_img[i] = img
log("Real picture has been prepared!")

fake_img = np.ndarray(shape=(n,c,h,w))
for i in range(n):
    img,org_img = prepare_pic(fake_pic)
    fake_img[i] = img
log("Fake picture has been prepared!")


#Start sync inference
log("Start sync inference")
t0 = time()
val_ret = exec_net.infer(inputs={input_blob:val_img})
real_ret = exec_net.infer(inputs={input_blob:real_img})
fake_ret = exec_net.infer(inputs={input_blob:fake_img})
infer_time = (time() - t0) * 1000
log("End Sync inference, time: %fms" % infer_time)

def post_deal_ret(res):
    res = res[output_blob]
    res = np.squeeze(res)
    return res

def rsv_ret(fpath, ret):
    with open(fpath, "wb+") as f:
        pickle.dump(ret, f)
    return

def cal_con_dis(a, b):
    ret = np.dot(a,b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    cos = ret/(na*nb)
    return cos

val_ret = post_deal_ret(val_ret)
real_ret = post_deal_ret(real_ret)
fake_ret = post_deal_ret(fake_ret)
#rsv_ret("val_pic.pkl", val_ret)
#rsv_ret("real_pic.pkl", real_ret)
#rsv_ret("fake_pic.pkl", fake_ret)
exit(0)

#Process result
res = res[output_blob]
res = np.squeeze(res)

#Get faces that its confidence > confidence_tolerance
face_num = 0
for iD,lbl,conf,x0,y0,x1,y1 in res:
    if(conf > confidence_tolerance):
        label_text = "{:.1f}%".format(conf * 100)
        x0 = int(x0 * org_w)
        x1 = int(x1 * org_w)
        y0 = int(y0 * org_h)
        y1 = int(y1 * org_h)
        cv2.rectangle(org_img, (x0, y0), (x1, y1), box_color, box_thickness)
        cv2.putText(org_img, label_text, (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_text_color, 1)
        face_num += 1
    else:
        break
log("%d face%s detected!" %(face_num, ['', 's'][face_num > 1]))

cv2.imshow("Result", org_img)
if cv2.waitKey()&0xFF == ord('q'):
    cv2.destroyAllWindows()
    exit(0)
