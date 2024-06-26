import os

os.environ['MAVLINK20'] = '1'
os.environ['MAVLINK_DIALECT'] = 'ardupilotmega'

from pymavlink import mavutil
import threading
import time
import multiprocessing


from GTool import GTool
# PyMAVLink has an issue that received messages which contain strings
# cannot be resent, because they become Python strings (not bytestrings)
# This converts those messages so your code doesn't crash when
# you try to send the message again.
def fixMAVLinkMessageForForward(msg):
	msg_type = msg.get_type()

	if msg_type in ('PARAM_VALUE', 'PARAM_REQUEST_READ', 'PARAM_SET'):
		if type(msg.param_id) == str:
			msg.param_id = msg.param_id.encode()
	elif msg_type == 'STATUSTEXT':
		if type(msg.text) == str:
			msg.text = msg.text.encode()
	return msg
	
class MavManager(GTool):
	def __init__(self, toolbox):
        
		super().__init__(toolbox)
		self.gps_raw = {
                'time_usec': '0',
                'fix_type': '0',
                'lat': '0',
                'lon': '0',
                'alt': '0',
                'HDOP': '0',
                'VDOP': '0'
        }
		self.mav_connected = False
		self.GCS_connected = False
		self.FC_connected = False
		self.toolBox = toolbox
		self._conn = toolbox.child_conn
		self.thread_terminate = False
		self.gcs_conn = None
		self.vehicle_conn = None
		self.lock = threading.Lock() # lock for internect communication
		self.lock2 = threading.Lock() # lock for temporary data(self.data) access
		self.ip = "" # Ground Control Station(GCS) ip
		self.data = "" # data from Pixhawk, temporary store here, can be access by other thread
		self.loop = threading.Thread(target=self.loopFunction)
		self.loop.daemon = True
		self.loop.start()
		self.loop2 = threading.Thread(target=self.processLoop) # process data with processLoop to prevent timeout from main loop function
		self.loop2.daemon = True
		self.loop2.start()
		
		
	# connect to Ground Control Station(GCS) with udp
	def connectGCS(self, ip):
		self.lock.acquire()
		if self.ip != ip:
			self.ip = ip
			if self.gcs_conn != None:
				self.gcs_conn.close()			
			self.gcs_conn = mavutil.mavlink_connection(f'udp:{ip}:14450', input=False)
			self.GCS_connected = True
			print(f"MavManager: GCS connected to {ip}")
		self.lock.release()
		
	# connect to pixhawk board with usb
	def connectVehicle(self, dev):
		if self.vehicle_conn != None:
				self.vehicle_conn.close()
		self.vehicle_conn = mavutil.mavlink_connection(dev, baud=57600)
		self.FC_connected = True
		
		msg = self.vehicle_conn.mav.request_data_stream_encode(
                0,
                0,
				24, # massage ID
                1, # rate(Hz)
                1, # Turn on
				)
		self.vehicle_conn.mav.send(msg)
		print(f"MavManager: FC connected to {dev}")

	def loopFunction(self):
		while True:
			if self.thread_terminate is True:
				break
            # Don't block for a GCS message - we have messages
            # from the vehicle to get too
			if self.vehicle_conn != None:
				self.lock.acquire()
				vcl_msg = self.vehicle_conn.recv_match(blocking=False)
				
				gcs_msg_p = ''

				if self.gcs_conn != None:
					gcs_msg_p = self.gcs_conn.recv_match(blocking=False)
					self.handleMsg(vcl_msg, self.gcs_conn)
					self.handleMsg(gcs_msg_p, self.vehicle_conn)
				
				self.lock.release()
			# Don't abuse the CPU by running the loop at maximum speed
			time.sleep(0.0001)
	def handleMsg(self, msg, target):
		
		if msg is None:
			pass
		elif msg.get_type == '':
			print("*** Fatal MavManager: Mavlink_message base type")
			pass
		elif msg.get_type() != 'BAD_DATA':
			#For debug
			
			if msg.get_type() == 'HEARTBEAT':
				self.lock2.acquire()
				self.data ='HEARTBEAT'
				self.lock2.release()

			if msg.get_type() == 'GPS_RAW_INT':
				self.lock2.acquire()
				self.gps_raw['time_usec'] = msg.time_usec
				self.gps_raw['fix_type'] = msg.fix_type
				self.gps_raw['lon'] = msg.lon
				self.gps_raw['lat'] = msg.lat
				self.gps_raw['alt'] = msg.alt
				self.gps_raw['HDOP'] = msg.eph
				self.gps_raw['VDOP'] = msg.epv
				self.lock2.release()
				#print(f"GPS: time_usec:{msg.time_usec}, lat:{msg.lat}, lon:{msg.lon}, alt:{msg.alt}")
				#print(f"     fix_type:{msg.fix_type}, h_acc:{msg.h_acc}, v_acc:{msg.v_acc}")


			# We now have a message we want to forward. Now we need to
			# make it safe to send
			msg = fixMAVLinkMessageForForward(msg)
			# Finally, in order to forward this, we actually need to
			# hack PyMAVLink so the message has the right source
			# information attached.
			target.mav.srcSystem = msg.get_srcSystem()
			target.mav.srcComponent = msg.get_srcComponent()
	
			# Only now is it safe to send the message
			target.mav.send(msg)

	def processLoop(self):
		while True:
			self.lock2.acquire()
			
			out_msg = self.data
			self.data = ""
			self.lock2.release()
			
			if out_msg == 'HEARTBEAT':
				self._conn.send(out_msg)
				time.sleep(0.1)
				
	def gps_data(self):
		self.lock2.acquire()
		gdata = self.gps_raw
		self.lock2.release()
		return gdata
	
	def send_distance_sensor_data(self, direction = 0, d = 0):
		try:
			distance = int(d/10)  # 固定距离值，单位为厘米（5米 = 500厘米）
			min_distance = 20   # 最小检测距离，单位为厘米
			max_distance = 1000 # 最大检测距离，单位为厘米
			current_time = 0  # 当前时间，单位为毫秒
			sensor_type = 0  # 传感器类型
			sensor_id = 0  # 传感器ID
			orientation = direction  # 方向，0表示正前方
			covariance = 0  # 协方差，0表示测量无误差
			#print(distance)

			
			# 调用距离传感器编码函数
			msg = self.vehicle_conn.mav.distance_sensor_encode(
				current_time,
				min_distance,
				max_distance,
				distance,
				sensor_type,
				sensor_id,
				orientation,
				covariance,
				)

			# 发送消息
			self.vehicle_conn.mav.send(msg)
			#print("Distance data sent")
		except Exception as e:
			print(f"Error sending distance data: {e}")



