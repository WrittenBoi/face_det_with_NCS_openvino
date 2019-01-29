#!/usr/bin/env python3
#encoding=utf-8
import sys,os,cv2
from time import time
import numpy as np, math
from argparse import ArgumentParser
from openvino.inference_engine import IENetwork, IEPlugin

class faceDet():
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
        faces = [e[2:] for e in res if e[2] >= self.threshold]
        return faces
