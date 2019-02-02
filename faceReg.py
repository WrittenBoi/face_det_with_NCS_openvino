#!/usr/bin/env python3
#encoding=utf-8
import sys,os,cv2
from time import time
import numpy as np, math
from configparser import ConfigParser
from openvino.inference_engine import IENetwork, IEPlugin

CONFIG_FILE = r"./config"
DEBUG = False

class faceReg():
    def __init__(self, device, model, weights, threshold, cpu_ext):
        self.device = device
        self.model = model
        self.weights = weights
        self.threshold = threshold
        self.plugin = IEPlugin(device=device)
        if "CPU" == device:
            self.plugin.add_cpu_extension(cpu_ext)
        self.net = IENetwork(model=model, weights=weights)
        self.input_blob = next(iter(self.net.inputs))
        self.output_blob = next(iter(self.net.outputs))
        self.n,self.c,self.h,self.w = self.net.inputs[self.input_blob].shape

        t0 = time()
        self.exec_net = self.plugin.load(network=self.net)
        self.load_time = (time() - t0) * 1000

        self.last_infer_time = 0.0

    def __str__(self):
        device = "Inference device: %s" % self.device
        model = "Model: %s" % self.model
        weights = "Weights: %s" % self.weights
        iShape = "Input shape: %s" % self.net.inputs[self.input_blob].shape
        oShape = "Output shape: %s" % self.net.outputs[self.output_blob].shape
        ldTime = "Model load time: %f ms" % self.load_time
        ifTime = "Last infer time: %f ms" % self.last_infer_time
        return '\n'.join([device, model, weights, iShape,
                    oShape, ldTime, ifTime])

    def __prepare_img(self, input_img):
        images = np.ndarray(shape=(self.n,self.c,self.h,self.w))
        for i in range(self.n):
            img = cv2.resize(input_img,(self.w, self.h))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.transpose((2,0,1)) # Change data layout from HWC to CHW
            images[i] = img
        return images

    def predict_sync(self, image):
        #Prepare image
        images = self.__prepare_img(image)

        t0 = time()
        res = self.exec_net.infer(inputs={self.input_blob: images})
        infer_time = (time() - t0) * 1000
        self.last_infer_time = infer_time

        #Process reuslt
        res = np.squeeze(res[self.output_blob])
        return res

    def predict_async(self, image):
        #Prepare image
        images = self.__prepare_img(image)
        #Send infer req
        self.last_infer_time = time()
        handle = self.exec_net.start_async(request_id=0,
                                inputs={self.input_blob: images})
        return handle

    def get_ret_async(self):
        ret = False
        res = []

        #State code:
        #  OK = 0
        #  GENERAL_ERROR = -1
        #  NOT_IMPLEMENTED = -2
        #  NETWORK_NOT_LOADED = -3
        #  PARAMETER_MISMATCH = -4
        #  NOT_FOUND = -5
        #  OUT_OF_BOUNDS = -6
        #  UNEXPECTED = -7
        #  REQUEST_BUSY = -8
        #  RESULT_NOT_READY = -9
        #  NOT_ALLOCATED = -10
        #  INFER_NOT_STARTED = -11
        #  NETWORK_NOT_READ = -12
        stat = self.exec_net.requests[0].wait(0)
        if(stat == 0):
            #get result error fixing
            stat = self.exec_net.requests[0].wait(-1)
            self.last_infer_time = time() - self.last_infer_time
            #fetch reuslt
            res = self.exec_net.requests[0].outputs
            #Process reuslt
            res = np.squeeze(res[self.output_blob])
            ret = True

        return ret,res

    def cal_similarity(self, face1, face2):
        #Cal cosine distance
        tmp = np.dot(face1, face2)
        na = np.linalg.norm(face1)
        nb = np.linalg.norm(face2)
        cos = tmp / (na * nb)

        #normalization result
        #cos = cos * 0.5 + 0.5
        return cos

def main():
    #Get config
    config = ConfigParser()
    config.read(CONFIG_FILE)

    #Fetch global config
    gbl_conf = config['GBL']
    global DEBUG
    DEBUG = eval(gbl_conf['debug'])
    dp_h = eval(gbl_conf['disp_wind_h'])
    dp_w = eval(gbl_conf['disp_wind_w'])
    box_color = eval(gbl_conf['box_color'])
    box_thickness = eval(gbl_conf['box_thickness'])
    label_text_color = eval(gbl_conf['label_text_color'])
    label_background_color = eval(gbl_conf['label_background_color'])

    ##Fetch face detection config
    #fd_conf = config['FD']
    #fd_device = fd_conf['device']
    #fd_model = fd_conf['model']
    #fd_weights = fd_conf['weights']
    #fd_threshold = float(fd_conf['threshold'])
    #fd_cpu_ext = fd_conf['cpu_ext']

    #Fetch face recognition config
    fr_conf = config['FR']
    fr_device = fr_conf['device']
    fr_model = fr_conf['model']
    fr_weights = fr_conf['weights']
    fr_threshold = float(fr_conf['threshold'])
    fr_cpu_ext = fr_conf['cpu_ext']
    fr_std_face = fr_conf['std_face']
    fr_match_dir = fr_conf['match_dir']

    #Face recognition init
    face_reg_obj = faceReg(fr_device, fr_model, fr_weights,
                            fr_threshold, fr_cpu_ext)

    #Infer std face
    img = cv2.imread(fr_std_face)
    std_face = face_reg_obj.predict_sync(img)

    #Match each face
    match_list = os.listdir(fr_match_dir)
    for e in match_list:
        #Get image
        img = cv2.imread(os.path.join(fr_match_dir, e))

        #Start inference
        #face = face_reg_obj.predict_sync(img)
        face_reg_obj.predict_async(img)
        ret,face = face_reg_obj.get_ret_async()
        while(not ret):
            ret,face = face_reg_obj.get_ret_async()

        #Cal similarity
        sim = face_reg_obj.cal_similarity(std_face, face)
        reg_time = face_reg_obj.last_infer_time

        #Show result
        print("%s:\tsimilarity %.2f in %.2f ms" % (e, sim, reg_time * 1000))
    return

if __name__ == "__main__":
    main()
    exit(0)
