import gi
import subprocess
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib, GObject

import GToolBox
from GTool import GTool
import VideoFormat

class VideoManager(GTool):
	def __init__(self, toolbox):
		super().__init__(toolbox)
		self.sys = 'buster'


		self.pipelinesexist = []
		self.pipelines = []
		self.pipelines_state = []
		self.camera_format = []
		self.videoFormatList = {}
		
		if(self._toolBox.OS == 'bionic'): # for Ubuntu 18.04 (Jetson Nano
				self.get_video_format_Ubuntu_18_04()
		else:
			self.get_video_format()

		self.portOccupied = {} # {port, videoNo}

		GObject.threads_init()
		Gst.init(None)

		self.createPipelines()
		print("[o] VideoManager: started")
		index = 0
		if len(self.videoFormatList) == 0:
			print("      - no camera connected")
		for form in self.videoFormatList:
			for video in self.videoFormatList[form]:
				print(f"      - {index}:  video{video[0]} {video[1]}")
			index += 1
		#if self.toolBox().oakCam.hasCamera:
		#	print("      - OAK camera connected")

	def createPipelines(self):
		for i in range(0, 10):
			pipeline = Gst.Pipeline()
			self.pipelines.append(pipeline)
			self.pipelinesexist.append(i)
			self.pipelines_state.append(False)

	#get video format from existing camera devices
	def get_video_format(self):
		#Check camera device
		for i in range(0,10):
			newCamera = True
			try:
				cmd = "v4l2-ctl -d /dev/video{} --list-formats-ext".format(i)
				returned_value = subprocess.check_output(cmd,shell=True,stderr=subprocess.DEVNULL).replace(b'\t',b'').decode("utf-8")  # returns the exit code in unix
			except:
				continue
			line_list = returned_value.splitlines()
			new_line_list = list()
			for j in line_list:
				if len(j.split()) == 0:
					continue
				elif j.split()[0][0] =='[':
					form = j.split()[1][1:-1]
				elif j.split()[0] =='Size:':
					size = j.split()[2]
					width, height = size.split('x')
				elif j.split()[0] == 'Interval:':
					fps = j.split()[3][1:].split('.')[0]
					self.camera_format.append('video{} {} width={} height={} framerate={}'.format(i,form, width, height , fps))
					index = self._toolBox.config.getVideoFormatIndex(width,height,fps)
					if index != -1:
						if index not in self.videoFormatList:
							self.videoFormatList[index] = []
							self.videoFormatList[index].append([i,form])
						else:
							video_index = 0
							add = True
							for video in self.videoFormatList[index]:
								if video[0] == i:
									if form =='MJPG':
										self.videoFormatList[index].pop(video_index)
										self.videoFormatList[index].append([i,form])
										add = False
									else:
										add = False
										break
								video_index += 1
							if add == True:
								self.videoFormatList[index].append([i,form])
						
	def get_video_format_Ubuntu_18_04(self):	
		#Check camera device
		for i in range(0,10):
			newCamera = True
			try:
				cmd = "v4l2-ctl -d /dev/video{} --list-formats-ext".format(i)
				returned_value = subprocess.check_output(cmd,shell=True,stderr=subprocess.DEVNULL).replace(b'\t',b'').decode("utf-8")  # returns the exit code in unix
			except:
				continue
			line_list = returned_value.splitlines()
			new_line_list = list()
			for j in line_list:
				# print(j.split())
				if len(j.split()) == 0:
					continue
				elif j.split()[0] =='Pixel':
					form = j.split()[2][1:-1]
				elif j.split()[0] =='Size:':
					size = j.split()[2]
					width, height = size.split('x')
				elif j.split()[0] == 'Interval:':
					fps = j.split()[3][1:].split('.')[0]
					self.camera_format.append('video{} {} width={} height={} framerate={}'.format(i,form, width, height , fps))
					index = self._toolBox.config.getVideoFormatIndex(width, height, fps)
					if index != -1:
						if index not in self.videoFormatList:
							self.videoFormatList[index] = []
							self.videoFormatList[index].append([i,form])
						else:
							video_index = 0
							add = True
							for video in self.videoFormatList[index]:
								if video[0] == i:
									if form =='MJPG':
										self.videoFormatList[index].pop(video_index)
										self.videoFormatList[index].append([i,form])
										add = False
									else:
										add = False
										break
								video_index += 1
							if add == True:
								self.videoFormatList[index].append([i,form])

	def play(self, cam, format, width, height, framerate, encoder, IP, port):
		gstring = VideoFormat.getFormatCMD(self._toolBox.OS, cam, format, width, height, framerate, encoder, IP, port)
		print(gstring)
		if port in self.portOccupied:
			videoToStop = self.portOccupied[port]
			self.pipelines[videoToStop].set_state(Gst.State.NULL)
			self.pipelines_state[videoToStop] = False
			print("  -quit occupied: video"+str(videoToStop))

		self.portOccupied[port] = cam
		if self.pipelines_state[cam] == True:
			self.pipelines[cam].set_state(Gst.State.NULL)
			self.pipelines[cam] = Gst.parse_launch(gstring)
			self.pipelines[cam].set_state(Gst.State.PLAYING)

		else:
			self.pipelines[cam] = Gst.parse_launch(gstring)
			self.pipelines[cam].set_state(Gst.State.PLAYING)
			self.pipelines_state[cam] = True
	def stop(self, video):
		if self.pipelines_state[video] == True:
			self.pipelines[video].set_state(Gst.State.NULL)
			self.pipelines_state[video] = False
