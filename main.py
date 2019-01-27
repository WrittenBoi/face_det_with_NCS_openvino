import sys,os,cv2
from time import time
import numpy as np, math
from argparse import ArgumentParser
from openvino.inference_engine import IENetwork, IEPlugin

DEBUG = True
cpu_ext_path =  "lib/libcpu_extension.so"

def log(*args):
    if(DEBUG):
        print(*args)

class faceDet():
    def __init__(self, device, model, weights):
        self.device = device
        self.model = model
        self.weights = weights
        self.exec_net = None

    def __mod_init(self):
        #Plugin init
        plugin = IEPlugin(device=self.device)
        if "CPU" == self.device:
            plugin.add_cpu_extension(cpu_ext_path)
        log("Plugin %s init OK!" % self.device)

        #Read IR
        if "MYRIAD" == self.device:
            self.model = self.model.replace("FP32", "FP16")
            self.weights = self.weights.replace("FP32", "FP16")
        net = IENetwork(model=self.model, weights=self.weights)
        log("Loading network file:\n\t{}\n\t{}".format(self.model, self.weights))

        #Get I/O blobs
        self.input_blob = next(iter(net.inputs))
        self.output_blob = next(iter(net.outputs))
        log("Input batch size:", net.batch_size)
        log("Input Shape:", net.inputs[self.input_blob].shape)
        log("Output Shape:", net.outputs[self.output_blob].shape)
        self.n,self.c,self.h,self.w = net.inputs[input_blob].shape

        #Load model to the plugin
        t0 = time()
        self.exec_net = plugin.load(network=net)
        load_time = (time() - t0) * 1000
        log("Load model to the plugin, time: %fms" % load_time)
        del net
