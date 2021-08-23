import numpy as np
import math

class Point:
    def __init__(self,x_init,y_init):
        self.x = x_init
        self.y = y_init

    def shift(self, x, y):
        self.x += x
        self.y += y

    def __repr__(self):
        return "".join(["Point(", str(self.x), ",", str(self.y), ")"])

def getNewPoint(origen, angle, distancia):
  new_x = int(round(origen.x + distancia*math.cos(angle+math.pi/2)))
  new_y = int(round(origen.y + distancia*math.sin(angle+math.pi/2)))
  return Point(new_x, new_y)

def getPointList(p1, p2):
  point_list = []

  p1 = np.array([p1.y,p1.x])
  p2 = np.array([p2.y,p2.x])

  p = p1
  d = p2-p1
  N = np.max(np.abs(d))
  s = d/N
  #print(np.rint(p).astype('int'))
  point_list.append(Point(int(round(p[1])),int(round(p[0]))))
  for ii in range(0,N):
    p = p+s;
    #print(np.rint(p).astype('int'))
    point_list.append(Point(int(round(p[1])),int(round(p[0]))))

  return point_list

def checkLinea(point_list, bin_img):
  linea_incompleta = False
  for punto in point_list:
    linea_incompleta = (punto.x < 0
            or punto.x > 255
            or punto.y < 0
            or punto.y > 255
            or bin_img[punto.y, punto.x] == 0)
    if linea_incompleta:
      break
  return linea_incompleta

def esLineaCompleta(point_list, bin_img):
  completa = True
  w = bin_img.shape[1]-1
  h = bin_img.shape[0]-1
  for point in point_list:
    completa = (point.x >= 0
                and point.x <= w
                and point.y >= 0
                and point.y <= h
                and bin_img[point.y, point.x] > 0)
    if not completa:
      break
  return completa