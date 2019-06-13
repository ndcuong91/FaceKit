#!/usr/bin/python3
from ctypes import *
import cv2
import numpy as np
import sys
import os
import time
from ipdb import set_trace as dbg
from enum import IntEnum
import imutils

tracking=False
lib_dir='/home/atsg/PycharmProjects/face_recognition/FaceKit/PCN/'

class CPoint(Structure):
    _fields_ = [("x", c_int),
                ("y", c_int)]

FEAT_POINTS = 14
class CWindow(Structure):
    _fields_ = [("x", c_int),
                ("y", c_int),
                ("width", c_int),
                ("angle", c_int),
                ("score", c_float),
                ("points",CPoint*FEAT_POINTS)]

class FeatEnam(IntEnum):
    CHIN_0 = 0
    CHIN_1 = 1
    CHIN_2 = 2
    CHIN_3 = 3
    CHIN_4 = 4
    CHIN_5 = 5
    CHIN_6 = 6
    CHIN_7 = 7
    CHIN_8 = 8
    NOSE = 9
    EYE_LEFT = 10
    EYE_RIGHT = 11
    MOUTH_LEFT = 12
    MOUTH_RIGHT = 13
    FEAT_POINTS = 14

if (tracking):
    lib = CDLL(os.path.join(lib_dir,'libPCN_tracking.so'))
else:
    lib = CDLL(os.path.join(lib_dir,'libPCN_no_tracking.so'))

init_detector = lib.init_detector
#void *init_detector(const char *detection_model_path, 
#            const char *pcn1_proto, const char *pcn2_proto, const char *pcn3_proto, 
#            const char *tracking_model_path, const char *tracking_proto,
#            int min_face_size, float pyramid_scale_factor, float detection_thresh_stage1,
#            float detection_thresh_stage2, float detection_thresh_stage3, int tracking_period,
#            float tracking_thresh, int do_smooth)
init_detector.argtypes = [
        c_char_p, c_char_p, c_char_p, 
        c_char_p, c_char_p, c_char_p,
        c_int,c_float,c_float,c_float,
        c_float,c_int,c_float,c_int]
init_detector.restype = c_void_p

#CWindow* detect_faces(void* pcn, unsigned char* raw_img,size_t rows, size_t cols, int *lwin)
detect_faces = lib.detect_faces
detect_faces.argtypes = [c_void_p, POINTER(c_ubyte),c_size_t,c_size_t,POINTER(c_int)]
detect_faces.restype = POINTER(CWindow)

#void free_faces(CWindow* wins)
free_faces = lib.free_faces
free_faces.argtypes= [c_void_p]

# void free_detector(void *pcn)
free_detector = lib.free_detector
free_detector.argtypes= [c_void_p]

CYAN=(255,255,0)
BLUE=(255,0,0)
RED=(0,0,255)
GREEN=(0,255,0)
YELLOW=(0,255,255)


def get_list_dir_in_folder(dir):
    sub_dir = [o for o in os.listdir(dir) if os.path.isdir(os.path.join(dir, o))]
    return sub_dir


def get_list_file_in_folder(dir, ext='jpg'):
    included_extensions = [ext]
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names

def DrawFace(win,img):
    width = 2
    x1 = win.x
    y1 = win.y
    x2 = win.width + win.x - 1
    y2 = win.width + win.y - 1
    centerX = (x1 + x2) / 2
    centerY = (y1 + y2) / 2
    angle = win.angle
    R = cv2.getRotationMatrix2D((centerX,centerY),angle,1)
    pts = np.array([[x1,y1,1],[x1,y2,1],[x2,y2,1],[x2,y1,1]], np.int32)
    pts = (pts @ R.T).astype(int) #Rotate points
    pts = pts.reshape((-1,1,2))
    cv2.polylines(img,[pts],True,CYAN,width)
    cv2.line(img, (pts[0][0][0],pts[0][0][1]), (pts[3][0][0],pts[3][0][1]), BLUE, width)
  
def DrawPoints(win,img):
    width = 3
    f = FeatEnam.NOSE
    cv2.circle(img,(win.points[f].x,win.points[f].y),width,GREEN,-1)
    f = FeatEnam.EYE_LEFT
    cv2.circle(img,(win.points[f].x,win.points[f].y),width,YELLOW,-1)
    f = FeatEnam.EYE_RIGHT
    cv2.circle(img,(win.points[f].x,win.points[f].y),width,YELLOW,-1)
    f = FeatEnam.MOUTH_LEFT
    cv2.circle(img,(win.points[f].x,win.points[f].y),width,RED,-1)
    f = FeatEnam.MOUTH_RIGHT
    cv2.circle(img,(win.points[f].x,win.points[f].y),width,RED,-1)
    for i in range(8):
        cv2.circle(img,(win.points[i].x,win.points[i].y),width,BLUE,-1)

def SetThreadCount(threads):
    os.environ['OMP_NUM_THREADS'] = str(threads)

def c_str(str_in):
    return c_char_p(str_in.encode('utf-8'))

def initialize():
    SetThreadCount(1)
    path = '/usr/local/share/pcn/'
    detection_model_path = c_str(path + "PCN.caffemodel")
    pcn1_proto = c_str(path + "PCN-1.prototxt")
    pcn2_proto = c_str(path + "PCN-2.prototxt")
    pcn3_proto = c_str(path + "PCN-3.prototxt")
    tracking_model_path = c_str(path + "PCN-Tracking.caffemodel")
    tracking_proto = c_str(path + "PCN-Tracking.prototxt")

    min_face_size=40 # minimum face size to detect >20
    image_pyramid_scale_factor=1.45 # scaleing factor of image pyramid [1.4;1.6]
    score_thres = (0.5, 0.5, 0.98) # score threshold of detected faces [0;1]
    smooth=0 #Smooth the face boxes or not (smooth = true or false, recommend using it on video to get stabler face boxes)
    tracking_period=30
    tracking_thres=0.9

    detector = init_detector(detection_model_path,
                             pcn1_proto, pcn2_proto, pcn3_proto,
                             tracking_model_path, tracking_proto,
                             min_face_size,
                             image_pyramid_scale_factor,
                             score_thres[0], score_thres[1], score_thres[2],
                             tracking_period, tracking_thres,
                             smooth)
    return detector

def detect_dir(detector, input_dir, rotate=0):
    print ()
    print ('Begin test ',input_dir,'with angle',rotate)
    output_dir = input_dir + '_result_'+str(rotate)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    outputNG_dir = output_dir + '/NG'
    if not os.path.exists(outputNG_dir):
        os.makedirs(outputNG_dir)

    list_file = get_list_file_in_folder(input_dir)
    list_file = sorted(list_file)

    total = 0
    detected = 0
    for img_file in list_file:
        frame = cv2.imread(os.path.join(input_dir, img_file))
        frame = imutils.rotate_bound(frame, rotate)
        width = frame.shape[1]
        height = frame.shape[0]
        face_count = c_int(0)
        raw_data = frame.ctypes.data_as(POINTER(c_ubyte))

        windows = detect_faces(detector, raw_data, int(height), int(width), pointer(face_count))

        num_face = face_count.value
        for i in range(num_face):
            DrawFace(windows[i], frame)
            DrawPoints(windows[i], frame)
        free_faces(windows)
        total += 1

        resized = cv2.resize(frame, (int(width/2), int(height/2)), interpolation=cv2.INTER_CUBIC)

        if (num_face < 1):
            print('NG:', img_file, '---------------------------------')
            cv2.imwrite(os.path.join(outputNG_dir, img_file), resized)
        else:
            #print('OK:', img_file)
            cv2.imwrite(os.path.join(output_dir,img_file),resized)
            detected += 1
        # cv2.imshow('PCN', frame)
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     break

    print(rotate, ', Detected:', detected, ', Total:', total, ', Accuracy:', float(detected) / float(total))

    import shutil
    # shutil.copy('/home/atsg/PycharmProjects/face_recognition/FaceKit/PCN/PyPCN_new.py', os.path.join(output_dir,'PyPCN_New.py'))

def detect_cam(detector):
    if len(sys.argv)==2:
        cap = cv2.VideoCapture(sys.argv[1])
    else:
        cap = cv2.VideoCapture(0)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    while cap.isOpened():
        ret, frame = cap.read()
        if frame.shape[0] == 0:
            break
        start = time.time()
        face_count = c_int(0)
        raw_data = frame.ctypes.data_as(POINTER(c_ubyte))

        windows = detect_faces(detector, raw_data,
                               int(height), int(width),
                               pointer(face_count))
        end = time.time()
        for i in range(face_count.value):
            DrawFace(windows[i], frame)
            DrawPoints(windows[i], frame)
        free_faces(windows)
        fps = int(1 / (end - start))
        cv2.putText(frame, str(fps) + "fps", (20, 45), 4, 1, (0, 0, 125))
        cv2.imshow('PCN', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__=="__main__":
    detector = initialize()

    #detect_cam(detector)
    input_dir = '/home/atsg/PycharmProjects/face_recognition/tiepnh/OK'

    for i in range(0,360,45):
        detect_dir(detector, input_dir, rotate=i)

    free_detector(detector)

