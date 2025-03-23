from models import TRTModule  # isort:skip
import argparse
from pathlib import Path
import threading
import time
import multiprocessing 
import math

import cv2
import torch
import numpy as np

from config import CLASSES, COLORS
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox, path_to_list
from GTool import GTool
from distance import distance, getR0

multiprocessing.set_start_method('spawn', force=True)


def detectTask(conn, input): # Thread that read data from oak camera
    enabled = True
    #connLoop = threading.Thread(target=connLoop)
    #connLoop.daemon = True
    #connLoop.start()
    cap_send = None
    out_send = None
    w = 0
    h = 0
    playing = False

    device = torch.device('cuda:0')
    Engine = TRTModule('yolov8s.engine', device)
    H, W = Engine.inp_info[0].shape[-2:]
    

    # set desired output names order
    Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])
    K0 = np.array(
        [[300, 0., 320],
        [0., 300, 240],
        [0., 0., 1.]]
    )
    cam_height = 0.3
    R0 = getR0(math.pi, 0)
    while True:
        if not playing:
            msg = input.get()
            if msg[0] == "p":
                msg = msg[1:]
                if cap_send != None:
                    cap_send.release()
                if out_send != None:
                    cap_send.release()
                video_pipeline = f'v4l2src device=/dev/video{msg[0]} ! video/x-raw, format=YUY2, width={msg[2]}, height={msg[3]}, framerate={msg[4]}/1 ! videoconvert ! appsink'
                cap_send = cv2.VideoCapture(video_pipeline, cv2.CAP_GSTREAMER)
                w = cap_send.get(cv2.CAP_PROP_FRAME_WIDTH)
                h = cap_send.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap_send.get(cv2.CAP_PROP_FPS)
                out_send = cv2.VideoWriter(f'appsrc ! videoconvert ! video/x-raw,format=I420 ! nvvideoconvert ! video/x-raw(memory:NVMM) ! nvv4l2h264enc ! rtph264pay pt=96 config-interval=1 ! udpsink host={msg[5]} port={msg[6]}'\
                    ,cv2.CAP_GSTREAMER\
                    ,0\
                    , int(fps)\
                    , (int(w), int(h))\
                    , True)

                if not cap_send.isOpened():
                    print('VideoCapture not opened')
                    continue
                if not out_send.isOpened():
                    print('VideoWriter not opened')
                    continue
                playing = True
            else:
                continue
        elif not input.empty():
            msg = input.get()
            if msg[0] == "x":
                if cap_send != None:
                    cap_send.release()
                if out_send != None:
                    cap_send.release()
                playing = False
                continue
            elif msg[0] == "i":
                pitch = math.pi/2 - float(msg[1])
                roll = - float(msg[2])
                R0 = getR0(pitch, roll)
                #print(R0)
            
        ret,frame = cap_send.read()
        if not ret:
            print('empty frame')
            break
        draw = frame.copy()
        frame, ratio, dwdh = letterbox(frame, (W, H))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb, return_seg=False)
        dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
        tensor = torch.asarray(tensor, device=device)
        # inference
        data = Engine(tensor)
        
        bboxes, scores, labels = det_postprocess(data)
        bboxes -= dwdh
        bboxes /= ratio
        detectionMetric = [
        #   label[0], x[1], y[2], width[3], height[4], dist[5]
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0
        ]
        count = 0
        for (bbox, score, label) in zip(bboxes, scores, labels):
            bbox = bbox.round().int().tolist()
            cls_id = int(label)
            cls = CLASSES[cls_id]
            color = COLORS[cls]
            x = bbox[0]
            y = bbox[1]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            d = distance(K0, R0, cam_height, x+w/2, y+h)
            cv2.rectangle(draw,tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
            cv2.putText(draw,
                        f'{cls} {d:.1f}m', (bbox[0], bbox[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, [225, 255, 255], thickness=2)
            
            if count < 10:
                detectionMetric[count*6] = cls_id
                detectionMetric[count*6+1] = x
                detectionMetric[count*6+2] = y
                detectionMetric[count*6+3] = w
                detectionMetric[count*6+4] = h
                detectionMetric[count*6+5] = 0
            count += 1
        
        
        if out_send.isOpened():
            out_send.write(draw)
        if conn.empty():
            conn.put(detectionMetric, block= False)

    out_send.release()
    cap_send.release()


class JetsonDetect(GTool):
    def __init__(self, toolbox):

        super().__init__(toolbox)
        self.out_conn = multiprocessing.Queue(1)
        self.in_conn = multiprocessing.Queue(1)

        self.enabled = True
        self.detectionMetric = [
            #   label[0], x[1], y[2], width[3], height[4], dist[5]
                0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0
        ]
    def play(self, msg):
        msg.insert(0, "p")
        self.in_conn.put(msg)
    def stop(self):
        self.in_conn.put(["x"])
    def updateIMU(self, msg): #[pitch, roll]
        msg.insert(0, "i")
        self.in_conn.put(msg)
    def sendMsg(self, msg):
        self.in_conn.put(msg)
         
		# 啟動process，告訴作業系統幫你創建一個process，是Async的
    def getConn(self):
        return self.out_conn

    def startLoop(self):
        
        self.p = multiprocessing.Process(target = detectTask, args = (self.out_conn, self.in_conn))
        self.p.start()
        #self.outputLoop = threading.Thread(target=self.OutputLoop)
        #self.outputLoop.daemon = True
        #self.outputLoop.start()

    def OutputLoop(self): # Thread that send data to the networkmanager
        while True:
            distances = 0
            #self.toolBox.mavManager.send_distance_sensor_data(7, int(min(distances[:3])))
            #self.toolBox.mavManager.send_distance_sensor_data(0, int(min(distances[3:6])))
            #self.toolBox.mavManager.send_distance_sensor_data(1, int(min(distances[6:])))
            d = self.out_conn.get()
            #print(d)
            ##### TODO self.toolBox().sensorManager.send_detection_result(d)
            time.sleep(0.1)
    def exitProcess(self):
        self.out_conn.send('x')
    def connLoop(self):
        while True:
            self.conn.put(self.detectionMetric)
            time.sleep(0.05)
    def createPipeline(self, msg):
        if self.cap_send != None:
            self.cap_send.release()
        if self.out_send != None:
            self.cap_send.release()
        video_pipeline = f'v4l2src device=/dev/video{msg[0]} ! video/x-raw, format=YUY2, width={msg[2]}, height={msg[3]}, framerate={msg[4]}/1 ! videoconvert ! appsink'
        self.cap_send = cv2.VideoCapture(video_pipeline, cv2.CAP_GSTREAMER)
        self.w = self.cap_send.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.h = self.cap_send.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = self.cap_send.get(cv2.CAP_PROP_FPS)
        self.out_send = cv2.VideoWriter(f'appsrc ! videoconvert ! video/x-raw,format=I420 ! nvvideoconvert ! video/x-raw(memory:NVMM) ! nvv4l2h264enc ! rtph264pay pt=96 config-interval=1 ! udpsink host={msg[5]} port={msg[6]}'\
            ,cv2.CAP_GSTREAMER\
            ,0\
            , int(fps)\
            , (int(self.w), int(self.h))\
            , True)

    
            



if __name__ == '__main__':
    jd = JetsonDetect(0)
    jd.startLoop()
    input()
