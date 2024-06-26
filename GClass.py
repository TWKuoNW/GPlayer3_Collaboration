
HEARTBEAT = b'\x00'
FORMAT = b'\x01'
COMMAND = b'\x02'
QUIT = b'\x03'
SENSOR = b'\x04'

import serial
import anrot_module


class Peripheral:
	def __init__(self, idProduct = "", idVendor="", manufacturer="", dev="", ID = 0):
		self.idProduct = idProduct
		self.idVendor = idVendor
		# 未來加入bcdDevice 作為識別
		self.manufacturer = manufacturer
		self.dev = dev
		self.ID = ID
		self.IO = None
		self.m_IMU = None
		
		
	def getList(self):
		return [self.idProduct, self.idVendor, self.manufacturer, self.dev, self.ID]
	def connect(self):
		if self.manufacturer == "Arduino LLC":
			print("    -connected to arduino...")
			self.IO = serial.Serial(self.dev,9600)
		if self.manufacturer == "ArduPilot":
			print("    -connected to ardupilot...")

		if self.manufacturer == "Silicon Labs":
			print("    -connected to gyro...")
			self.m_IMU = anrot_module.anrot_module('./config.json')
			self.IO = 1
	def write(self, msg):
		if self.IO != None:
			if self.manufacturer == "Arduino LLC":
				self.IO.write((msg+"\n").encode())

	def read(self):
		if self.IO != None:
			if self.manufacturer == "Arduino LLC":
				return self.IO.readline().decode()[:-1]
			if self.manufacturer == "Silicon Labs":
				try:
					data = self.m_IMU.get_module_data(10)
					data = ", ".join(data).encode()
					
				except:
					print('error')
					data = b''
					pass
				return data.decode()
				

		else:
			return 0

	def __del__(self):
		if self.IO != None:
			self.IO.close()
		if self.m_IMU != None:
			self.m_IMU.close()

	

class Device:
	ID = 0
	#deviceName = ""
	type = 0
	settings = ""
	pinIDList = []
	dataBuffer = []

