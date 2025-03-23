from scipy.spatial.transform import Rotation
import math
import numpy as np

def distance(K,R,h,u,v):
  uv = np.array(
    [[u],
    [v],
    [1]]
  )

  Pc = np.matmul(np.linalg.inv(K), uv)
  Pw = np.matmul(np.linalg.inv(R), Pc)
  l = cam_height/Pw[-1]
  Pw2 = -Pw*(cam_height/Pw[-1])
  dist = np.linalg.norm(Pw2)
  #print(Pc)
  #print(Pw)
  #print(Pw2)
  return dist

def getR0(roll, pitch):
    return Rotation.from_euler('xyz', [pitch, 0, roll], degrees=True).as_matrix()