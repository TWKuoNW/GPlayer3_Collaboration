#!/usr/bin/env python3

import cv2
import depthai as dai
import math
import numpy as np
import threading
import time
from GTool import GTool

# Open up connection with oak-D S2 camera for depth detection

class OakCam(GTool):
    def __init__(self, toolBox):
        super().__init__(toolBox)
        self.distances = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.lock = threading.Lock() # lock for internect communication
        self.inputLoop = threading.Thread(target=self.InputLoop)
        self.hasCamera = False
        self.inputLoop.daemon = True
        self.inputLoop.start()
        self.outputLoop = threading.Thread(target=self.OutputLoop)
        self.outputLoop.daemon = True
        self.outputLoop.start()
        self.initialized = False
        timeout = 0
        while timeout<10:
            if not self.initialized:
                time.sleep(1)
                timeout += 1
            else:
                break
        
        
    def OutputLoop(self): # Thread that send data to the networkmanager
        while self.hasCamera:
            self.lock.acquire()
            distances = self.distances
            self.lock.release()
            self.toolBox().mavManager.send_distance_sensor_data(7, int(min(distances[:3])))
            self.toolBox().mavManager.send_distance_sensor_data(0, int(min(distances[3:6])))
            self.toolBox().mavManager.send_distance_sensor_data(1, int(min(distances[6:])))
            time.sleep(0.5)


    def InputLoop(self): # Thread that read data from oak camera
        out = cv2.VideoWriter(f'appsrc ! videoconvert ! omxh264enc ! rtph264pay pt=96 config-interval=1 ! udpsink host=192.168.0.99 port=5201'
        , cv2.CAP_GSTREAMER, 0, 30, (int(640),int(400)), True)
        out2 = cv2.VideoWriter(f'appsrc ! videoconvert ! omxh264enc ! rtph264pay pt=96 config-interval=1 ! udpsink host=192.168.0.99 port=5202'
        , cv2.CAP_GSTREAMER, 0, 30, (int(640),int(400)), True)
        # Create pipeline
        pipeline = dai.Pipeline()

        # Define sources and outputs
        camRgb = pipeline.create(dai.node.Camera)
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)
        spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

        rgbOut = pipeline.create(dai.node.XLinkOut)
        xoutDepth = pipeline.create(dai.node.XLinkOut)
        xoutSpatialData = pipeline.create(dai.node.XLinkOut)
        xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)

        rgbOut.setStreamName("rgb")
        xoutDepth.setStreamName("depth")
        xoutSpatialData.setStreamName("spatialData")
        xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

        # Properties

        camRgb.setPreviewSize(640, 400)
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setCamera("left")
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setCamera("right")

        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)
        spatialLocationCalculator.inputConfig.setWaitForMessage(False)

        # Create 10 ROIs
        for i in range(10):
            config = dai.SpatialLocationCalculatorConfigData()
            config.depthThresholds.lowerThreshold = 200
            config.depthThresholds.upperThreshold = 10000
            config.roi = dai.Rect(dai.Point2f(i*0.1, 0.45), dai.Point2f((i+1)*0.1, 0.55))
            spatialLocationCalculator.initialConfig.addROI(config)

        # Linking
        camRgb.preview.link(rgbOut.input)
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
        stereo.depth.link(spatialLocationCalculator.inputDepth)

        spatialLocationCalculator.out.link(xoutSpatialData.input)
        xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)
        try:
            # Connect to device and start pipeline
            with dai.Device(pipeline) as device:
                self.hasCamera = True
                device.setIrLaserDotProjectorBrightness(1000)

                # Output queue will be used to get the depth frames from the outputs defined above
                rgbQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
                depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
                spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
                color = (0,200,40)
                fontType = cv2.FONT_HERSHEY_TRIPLEX
                print(f"[o] OakCam: connected")
                self.initialized = True
                while True:
                    inRGB = rgbQueue.get()
                    inDepth = depthQueue.get() # Blocking call, will wait until a new data has arrived

                    rgbFrame = inRGB.getCvFrame()
                    depthFrame = inDepth.getFrame() # depthFrame values are in millimeters

                    depth_downscaled = depthFrame[::4]
                    if np.all(depth_downscaled == 0):
                        min_depth = 0  # Set a default minimum depth value when all elements are zero
                    else:
                        min_depth = np.percentile(depth_downscaled[depth_downscaled != 0], 1)
                    max_depth = np.percentile(depth_downscaled, 99)
                    depthFrameColor = np.interp(depthFrame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
                    depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

                    spatialData = spatialCalcQueue.get().getSpatialLocations()
                    direction = 0

                    
                    for depthData in spatialData:
                        roi = depthData.config.roi
                        roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])

                        xmin = int(roi.topLeft().x)
                        ymin = int(roi.topLeft().y)
                        xmax = int(roi.bottomRight().x)
                        ymax = int(roi.bottomRight().y)

                        coords = depthData.spatialCoordinates
                        distance = math.sqrt(coords.x ** 2 + coords.y ** 2 + coords.z ** 2)
                        
                        self.lock.acquire()
                        self.distances[direction] = distance
                        self.lock.release()
                        
                        direction = direction + 1
                        cv2.rectangle(rgbFrame, (xmin, ymin), (xmax, ymax), color, thickness=2)
                        cv2.putText(rgbFrame, "{:.1f}m".format(distance/1000), (xmin + 10, ymin + 20), fontType, 0.6, color)
                    # Show the frame
                    out.write(rgbFrame)
                    out2.write(depthFrameColor)
        except Exception as e:
            print(f"[*] OakCam: not connected: {e}")
            self.initialized = True
