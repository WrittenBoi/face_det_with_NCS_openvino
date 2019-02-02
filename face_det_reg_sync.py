#!/usr/bin/env python3
#encoding=utf-8

import sys,os,cv2
from time import time
import numpy as np, math
from configparser import ConfigParser
from openvino.inference_engine import IENetwork, IEPlugin
from faceDet import faceDet
from faceReg import faceReg

CONFIG_FILE = r"./config"
DEBUG = False

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

    #Fetch face recognition config
    fr_conf = config['FR']
    fr_device = fr_conf['device']
    fr_model = fr_conf['model']
    fr_weights = fr_conf['weights']
    fr_threshold = float(fr_conf['threshold'])
    fr_cpu_ext = fr_conf['cpu_ext']
    fr_std_face = fr_conf['std_face']
    fr_match_dir = fr_conf['match_dir']

    #Face detection Init
    face_det_obj = faceDet(fd_device, fd_model, fd_weights,
                            fd_threshold, fd_cpu_ext)

    #Face recognition init
    face_reg_obj = faceReg(fr_device, fr_model, fr_weights,
                            fr_threshold, fr_cpu_ext)

    #Camera init
    cap = cv2.VideoCapture(fd_cam_idx)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, dp_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, dp_h)

    #Infer std face
    img = cv2.imread(fr_std_face)
    std_face = face_reg_obj.predict_sync(img)

    #Main loop
    while(True):
        t0 = time()
        #Capture a pic
        ret, img = cap.read()

        t1 = time()
        #Detect faces
        faces = face_det_obj.predict_sync(img)

        t2 = time()
        face_sim_lst = []
        if(len(faces) > 0):
            #Recognition faces
            for conf,x0,y0,x1,y1 in faces:
                x0 = abs(int(x0 * dp_w))
                x1 = abs(int(x1 * dp_w))
                y0 = abs(int(y0 * dp_h))
                y1 = abs(int(y1 * dp_h))
                cur_face = face_reg_obj.predict_sync(img[y0:y1, x0:x1])
                sim = face_reg_obj.cal_similarity(std_face, cur_face)
                face_sim_lst.append(sim)
            #print("%d face detected!" % len(face_sim_lst))
            #print("Similarities:", face_sim_lst)
            max_sim = max(face_sim_lst)
            max_idx = face_sim_lst.index(max_sim)
            match_ret = faces[max_idx:max_idx+1]
            img = process_result(img, match_ret, box_color,
                            box_thickness, label_text_color)
        else:
            print("No face detected!")

        t3 = time()
        print("*" * 50)
        print("Image cap: %.2f ms" % ((t1 - t0) * 1000))
        print("Det time: %.2f ms" % ((t2 - t1) * 1000))
        print("Reg time: %.2f ms" % ((t3 - t2) * 1000))
        print("Total time: %.2f ms" % ((t3 - t0) * 1000))
        print("  %d face detected!" % len(face_sim_lst))
        print("  Similarities:", face_sim_lst)
        #Show result
        cv2.imshow("Result", img)

        #Wait key to continue or quit
        keycode = cv2.waitKey(1)
        if(keycode & 0xFF == ord('q')):
            break

if __name__ == "__main__":
    main()
    exit(0)
