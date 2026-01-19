import time
import serial
import struct
import logging
from Dev.Device import Device

# This device will connect to arduino, which canc accept command for winch control
# Command for arduino
            # stepper settings:          s,2,0,2 5 2000 1000 [operaton, header, ID, dirPin stepPin maxSpeed acceleration]
            # stepper control:           c,2,2,800 [operaton, header, ID, steps]
            # stepper stop:              z,2


CONTROL = b'\x05'

class WinchDevice(Device):
    def __init__(self, device_type, dev_path="", sensor_group_list = [], networkManager = None):
        super().__init__(device_type, dev_path, sensor_group_list, networkManager)
        self.isSerialInit = False
        self.control_type = 0
        try:
            self.serialOut = serial.Serial(port = self.dev_path, baudrate = 9600, timeout = 5) 
            self.isSerialInit = True
            # initialize winch on arduino
            self.send(f's,2000 1000')

        except serial.serialutil.SerialException: # if serial error
            logging.error("WinchDevice: Serial Error...")
            logging.info("WinchDevice: Trying to reconnect...")

        except Exception as e: # if other error
            logging.error(e) 
    
    if(PRE_BTN_STATE == 2 || remote_rundown == true){
      BTN_STATE = 0;
      remote_rundown = false;
      Serial.println("STOP !");
      OFFSET_P = stepper.currentPosition()+OFFSET_P;
      stepper.setCurrentPosition(0);
      u8g2.clearBuffer();
      u8g2.setFont(u8g2_font_logisoso16_tf); //設定字型
      u8g2.drawStr(0,16,String("Pos:"+String(STEPPER_CURRENT_P)).c_str());  //輸出文字
      //u8g2.updateDisplayArea(0, 0,16, 8);
      u8g2.sendBuffer();
    }
  }

  if(BTN_STATE == 0){ //run up 
    if(PRE_BTN_STATE == 0){
      stepper.run();
      
    }else{
      Serial.println("STOP !");
      OFFSET_P = stepper.currentPosition()+OFFSET_P;
      stepper.setCurrentPosition(0);
      u8g2.clearBuffer();
      u8g2.setFont(u8g2_font_logisoso16_tf); //設定字型
      u8g2.drawStr(0,16,String("Pos:"+String(STEPPER_CURRENT_P)).c_str());  //輸出文字
      //u8g2.updateDisplayArea(0, 0,16, 8);
      u8g2.sendBuffer();
    }
    
  }else if(BTN_STATE == 1){ //run up 
    if(PRE_BTN_STATE == 1){
      stepper.run();
    }else if(PRE_BTN_STATE == 2){
      Serial.println("STOP !");
      OFFSET_P = stepper.currentPosition()+OFFSET_P;
      stepper.setCurrentPosition(0);
      delay(500);
      Serial.println("RUN UP !");
      u8g2.clearBuffer();
      u8g2.setFont(u8g2_font_logisoso16_tf); //設定字型
      u8g2.drawStr(0,16,String("Pos: running").c_str());  //輸出文字
      u8g2.drawStr(0,40, "  RUN UP!!");
      u8g2.setFont(u8g2_font_open_iconic_arrow_2x_t);
      u8g2.setCursor(0,40);
      u8g2.drawGlyph(0, 40, 0x47);
      //u8g2.updateDisplayArea(0, 0,16, 8);
      u8g2.sendBuffer();
      stepper.move(-1000000);
    }else{
      Serial.println("RUN UP !");
      u8g2.clearBuffer();
      u8g2.setFont(u8g2_font_logisoso16_tf); //設定字型
      u8g2.drawStr(0,16,String("Pos: running").c_str());  //輸出文字
      u8g2.drawStr(0,40, "  RUN UP!!");
      u8g2.setFont(u8g2_font_open_iconic_arrow_2x_t);
      u8g2.setCursor(0,40);
      u8g2.drawGlyph(0, 40, 0x47);
      //u8g2.updateDisplayArea(0, 0,16, 8);
      u8g2.sendBuffer();
      stepper.move(-1000000);
    }
    
  }else if(BTN_STATE == 2){ //run down
    if(PRE_BTN_STATE == 2){
      stepper.run();
    }else if(PRE_BTN_STATE == 1){
      Serial.println("STOP !");
      OFFSET_P = stepper.currentPosition()+OFFSET_P;
      stepper.setCurrentPosition(0);
      delay(500);
      u8g2.clearBuffer();
      u8g2.setFont(u8g2_font_logisoso16_tf); //設定字型
      u8g2.drawStr(0,16,String("Pos: running").c_str());  //輸出文字
      u8g2.drawStr(0,40, "  RUN DOWN!!");
      u8g2.setFont(u8g2_font_open_iconic_arrow_2x_t);
      u8g2.setCursor(0,40);
      u8g2.drawGlyph(0, 40, 0x44);
      Serial.println("RUN DOWN !");
      u8g2.sendBuffer();
      stepper.move(1000000);
    }else{
      u8g2.clearBuffer();
      u8g2.setFont(u8g2_font_logisoso16_tf); //設定字型
      u8g2.drawStr(0,16,String("Pos: running").c_str());  //輸出文字
      u8g2.drawStr(0,40, "  RUN DOWN!!");
      u8g2.setFont(u8g2_font_open_iconic_arrow_2x_t);
      u8g2.setCursor(0,40);
      u8g2.drawGlyph(0, 40, 0x44);
      Serial.println("RUN DOWN !");
      u8g2.sendBuffer();
      //steppers.setSpeed(3000);
      stepper.move(1000000);
    }
    
    # setter
    def set(self):
        pass
        
    # process command for control
    def processCMD(self, control_type ,cmd):
        if self.isSerialInit == False:
            return
        if control_type == self.control_type:
            command_type = int(cmd[0])
            logging.info(f"control:{control_type}, command type:{command_type}, ")
            if command_type == 0:  # 讀取全部參數
                logging.info("  - set")
                # 待新增
            elif command_type == 1:  # 讀取部分參數
                pass
            elif command_type == 2:  # 寫入全部參數
                pass

            elif command_type == 3: #寫入部分參數
                index = int(cmd[1])
                logging.info(f"write index:{index}")
                if index == 0: #maxspeed
                    maxSpeed = int(struct.unpack("<I", cmd[2:])[0])
                    if maxSpeed>2000: #  maxspeed cant exceed 2000
                        pass
                    self.send(f's,{maxSpeed} {maxSpeed/2}')
                    logging.info(f"set maxspeed:{maxSpeed}")

            elif command_type == 4: #回傳全部參數
                pass
            elif command_type == 5: #回傳部分參數
                pass
            elif command_type == 6: #move
                step = int(struct.unpack("<i", cmd[1:])[0])
                logging.info(f"WinchDevice: move step {step}")
                if self.isSerialInit == True:
                    self.send(f'c,{step}')
            elif command_type == 7: #stop
                self.send(f'z,')
                logging.info("WinchDevice: stop")
            elif command_type == 8: # report step tension
                pass
            elif command_type == 9: # reset position
                self.send(f're')
                logging.info("WinchDevic: reset")

            
    def _io_loop(self):
        step = 0
        tension = 0
        status = 0
        while True:
            input = self.serialOut.readline()
            #logging.info(input)
            try:
                input = input.decode().split(",")
                if input[0] == "cs":
                    step = int(input[1])
                    tension = int(input[2])
                    if input[3][0] == 'S':
                        status = 0
                    elif input[3][0] == 'R':
                        status = 1
                    else:
                        status = 3
            except:
                pass
            time.sleep(0.2)
            data = struct.pack("<B", self.control_type)
            data += struct.pack("<B", 8)
            data += struct.pack("<i", step)
            data += struct.pack("<i", tension)
            data += struct.pack("<B", status)
            self.networkManager.sendMsg(b'\x05', data)

      
      if(field == 0){  
        //operation field
        // s : set data
        // c : control 
        // r : reset
        // z : stop, just for stepper motors
        operation = ldata;
        Serial.print("Operation:");
        Serial.println(ldata);
        if(operation == "re"){
          // reset tension and current position
          Serial.println("ack,re,0");
          stepper.setCurrentPosition(0);
          OFFSET_P = 0;
          WL = reading + 10000;
        }else if(ldata == "rq"){
          Serial.println("rs,0");
        }else if(ldata == "z"){
          //stop
          Serial.println("ack,z,0");
          OFFSET_P = stepper.currentPosition()+OFFSET_P;
          stepper.setCurrentPosition(0);
          //BTN_STATE = 0;
          
        }else if(ldata == "rp"){
          // reset current position
          stepper.setCurrentPosition(0);
          OFFSET_P = 0;
        }
      }else if(field == 1){
        if(operation == "s"){ //set
          int field2 = 0;
          String tmpdata = ldata;
          String ldata2;
          int index2 = 0;
          int pin1 = 0;  //dirPin or
          int pin2 = 0;  //setPin
          int maxSpeed = 0;
          int acc = 0;
          while(index2 != -1){
            index2 = split(ldata2, tmpdata, ' ');
            if(field2 == 0){
              Serial.print("Speed:");
              Serial.println(ldata2);//setMaxSpeed
              maxSpeed = ldata2.toInt();
            }else{
              Serial.print("Acc:");
              Serial.println(ldata2);//setMaxSpeed
              acc = ldata2.toInt();
              stepper.setAcceleration(acc);
              stepper.setMaxSpeed(maxSpeed);
            }
            field2++;
          }
        }else if(operation == "c"){
          //set move() position
          long steps = ldata.toInt()-STEPPER_CURRENT_P;
          if(steps>0){
            remote_rundown = true;
          }
          remote_controlling = true;
          stepper.move(steps);
          Serial.print("set move to:");
          Serial.println(steps);
        }else if(operation == "st"){
          //set tension
          long tension = ldata.toInt();
          WL = tension;
          Serial.print("set tension:");
          Serial.println(WL);
        }else if(operation == "t"){
          WL = ldata.toInt();
          Serial.print("WL:");
          Serial.println(WL);
        }
      }
      field ++;
    } 
  }
    long int t2 = millis();
}
