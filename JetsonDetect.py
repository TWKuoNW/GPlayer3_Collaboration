from models import TRTModule  # isort:skip
import argparse
from pathlib import Path
import threading
import time
import multiprocessing 


import cv2
import torch

from config import CLASSES, COLORS
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox, path_to_list
from GTool import GTool

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
    def sendMsg(self, msg):
        self.in_conn.put(msg)
         
		# 啟動process，告訴作業系統幫你創建一個process，是Async的
    def getConn(self):
        return self.out_conn

    def startLoop(self):
        
        self.p = multiprocessing.Process(target = self.InputLoop, args = (self.out_conn, self.in_conn))
        self.p.start()
        self.outputLoop = threading.Thread(target=self.OutputLoop)
        self.outputLoop.daemon = True
        self.outputLoop.start()

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

    def InputLoop(self, conn, input): # Thread that read data from oak camera
        self.enabled = True
        self.conn = conn
        self.connLoop = threading.Thread(target=self.connLoop)
        self.connLoop.daemon = True
        self.connLoop.start()
        self.cap_send = None
        self.out_send = None
        self.w = 0
        self.h = 0
        self.playing = False

        self.device = torch.device('cuda:0')
        self.Engine = TRTModule('yolov8s.engine', self.device)
        self.H, self.W = self.Engine.inp_info[0].shape[-2:]
        

        # set desired output names order
        self.Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])
        
        
        while True:
            if not self.playing:
                msg = input.get()
                if msg[0] == "p":
                    self.createPipeline(msg[1:])
                    if not self.cap_send.isOpened():
                        print('VideoCapture not opened')
                        continue
                    if not self.out_send.isOpened():
                        print('VideoWriter not opened')
                        continue
                    self.playing = True
                else:
                    continue
            elif not input.empty():
                msg = input.get()
                if msg[0] == "x":
                    if self.cap_send != None:
                        self.cap_send.release()
                    if self.out_send != None:
                        self.cap_send.release()
                    self.playing = False
                    continue
                
            ret,frame = self.cap_send.read()
            if not ret:
                print('empty frame')
                break
            draw = frame.copy()
            frame, ratio, dwdh = letterbox(frame, (self.W, self.H))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = blob(rgb, return_seg=False)
            dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=self.device)
            tensor = torch.asarray(tensor, device=self.device)
            # inference
            data = self.Engine(tensor)

            bboxes, scores, labels = det_postprocess(data)
            bboxes -= dwdh
            bboxes /= ratio
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
            count = 0
            for (bbox, score, label) in zip(bboxes, scores, labels):
                bbox = bbox.round().int().tolist()
                cls_id = int(label)
                cls = CLASSES[cls_id]
                color = COLORS[cls]
                cv2.rectangle(draw,tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
                cv2.putText(draw,
                            f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, [225, 255, 255], thickness=2)
                
                if count < 10:
                    self.detectionMetric[count*6] = cls_id
                    self.detectionMetric[count*6+1] = bbox[0]
                    self.detectionMetric[count*6+2] = bbox[1]
                    self.detectionMetric[count*6+3] = bbox[2] - bbox[0]
                    self.detectionMetric[count*6+4] = bbox[3] - bbox[1]
                    self.detectionMetric[count*6+5] = 0
                count += 1
            
            
            if self.out_send.isOpened():
                self.out_send.write(draw)
            if cv2.waitKey(1)&0xFF == ord('q'):
                break
        self.out_send.release()
        self.cap_send.release()
            



if __name__ == '__main__':
    jd = JetsonDetect(0)
    jd.startLoop()
    input()
