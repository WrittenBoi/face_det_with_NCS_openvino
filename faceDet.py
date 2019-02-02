#!/usr/bin/env python3
#encoding=utf-8
import sys,os,cv2
from time import time
import numpy as np, math
from configparser import ConfigParser
from openvino.inference_engine import IENetwork, IEPlugin

CONFIG_FILE = r"./config"
DEBUG = False

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
        #Result format: imageID,label,confidence,xMIN,yMIN,xMAX,yMAX
        faces = [e[2:] for e in res if e[2] >= self.threshold]
        return faces

    def predict_async(self, image):

        self.last_infer_time = time()

        #Prepare image
        images = self.__prepare_img(image)
        #Send infer req
        handle = self.exec_net.start_async(request_id=0,
                            inputs={self.input_blob: images})
        return handle

    def get_ret_async(self):
        faces = None

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
            #fetch reuslt
            res = self.exec_net.requests[0].outputs
            #Process reuslt
            res = np.squeeze(res[self.output_blob])
            #Result format: imageID,label,confidence,xMIN,yMIN,xMAX,yMAX
            faces = [e[2:] for e in res if e[2] >= self.threshold]

            self.last_infer_time = (time() - self.last_infer_time) * 1000

        return faces

def process_result(img, face_lst, box_color, box_thik, label_color):
    h,w,c = img.shape
    for conf,x0,y0,x1,y1 in face_lst:
        label_text = "{:.1f}%".format(conf * 100)
        x0 = int(x0 * w)
        x1 = int(x1 * w)
        y0 = int(y0 * h)
        y1 = int(y1 * h)
        cv2.rectangle(img, (x0, y0), (x1, y1), box_color, box_thik)
        cv2.putText(img, label_text, (x0, y0 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 1)
    return img

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

    #Fetch face detection config
    fd_conf = config['FD']
    fd_device = fd_conf['device']
    fd_model = fd_conf['model']
    fd_weights = fd_conf['weights']
    fd_threshold = float(fd_conf['threshold'])
    fd_cpu_ext = fd_conf['cpu_ext']
    fd_test_src = fd_conf['test_src']
    fd_test_dir = fd_conf['test_dir']
    fd_cam_idx = int(fd_conf['cam_idx'])

    #Face detection Init
    face_det_obj = faceDet(fd_device, fd_model, fd_weights,
                            fd_threshold, fd_cpu_ext)
    cv2.namedWindow("Result",0);
    cv2.resizeWindow("Result", dp_w, dp_h);
    if(fd_test_src == 'pic'):
        #Infer each test pic
        pic_list = os.listdir(fd_test_dir)
        for e in pic_list:
            img = cv2.imread(os.path.join(fd_test_dir, e))

            #Start inference
            #faces = face_det_obj.predict_sync(img)
            face_det_obj.predict_async(img)
            faces = face_det_obj.get_ret_async()
            while(faces == None):
                faces = face_det_obj.get_ret_async()

            #Process result
            img = cv2.resize(img, (dp_w, dp_h))
            img = process_result(img, faces, box_color,
                            box_thickness, label_text_color)

            #Add infer time label
            time_label = "DT: %.2fms" % face_det_obj.last_infer_time
            cv2.putText(img, time_label, (0, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_text_color, 1)
            cv2.imshow("Result", img)

            #Wait key to continue or quit
            keycode = cv2.waitKey()
            if(keycode & 0xFF == ord('q')):
                break
            else:
                continue

    else:
        cap = cv2.VideoCapture(fd_cam_idx)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, dp_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, dp_h)
        t0 = time()
        while(True):
            ret, img = cap.read()
            #Start inference
            #faces = face_det_obj.predict_sync(img)
            face_det_obj.predict_async(img)
            faces = face_det_obj.get_ret_async()
            while(faces == None):
                faces = face_det_obj.get_ret_async()

            #Process result
            img = cv2.resize(img, (dp_w, dp_h))
            img = process_result(img, faces, box_color,
                            box_thickness, label_text_color)
            t1 = time()
            fps = 1 / (t1 - t0)
            t0 = t1

            #Add infer time label
            time_label = "FPS: %.2f" % fps
            cv2.putText(img, time_label, (0, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_text_color, 1)
            cv2.imshow("Result", img)

            #Wait key to continue or quit
            keycode = cv2.waitKey(1)
            if(keycode & 0xFF == ord('q')):
                break

    cv2.destroyAllWindows()
    return

if __name__ == "__main__":
    main()
    exit(0)
