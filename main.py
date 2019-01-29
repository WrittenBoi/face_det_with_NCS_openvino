#!/usr/bin/env python3
#encoding=utf-8
import sys,os,cv2
from time import time
from configparser import ConfigParser
from faceDet import faceDet

CONFIG_FILE = r"./config"
DEBUG = False
def log(*args):
    if(DEBUG):
        print(*args)

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
    test_src = gbl_conf['test_src']
    test_dir = gbl_conf['test_dir']
    cam_idx = int(gbl_conf['cam_idx'])

    #Fetch face detection config
    fd_conf = config['FD']
    fd_device = fd_conf['device']
    fd_model = fd_conf['model']
    fd_weights = fd_conf['weights']
    fd_threshold = float(fd_conf['threshold'])
    fd_cpu_ext = fd_conf['cpu_ext']

    #Face detection Init
    face_det_obj = faceDet(fd_device, fd_model, fd_weights,
                            fd_threshold, fd_cpu_ext)
    cv2.namedWindow("Result",0);
    cv2.resizeWindow("Result", dp_w, dp_h);
    if(test_src == 'pic'):
        #Infer each test pic
        pic_list = os.listdir(test_dir)
        for e in pic_list:
            img = cv2.imread(os.path.join(test_dir, e))

            #Start inference
            faces = face_det_obj.predict_sync(img)

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
        cap = cv2.VideoCapture(cam_idx)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, dp_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, dp_h)
        t0 = time()
        while(True):
            ret, img = cap.read()
            #Start inference
            faces = face_det_obj.predict_sync(img)

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
