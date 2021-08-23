import mahotas as mh
import numpy as np
import math

def branchedPoints(skel):
    branch1 = np.array([[2, 1, 2], [1, 1, 1], [2, 2, 2]])
    branch2 = np.array([[1, 2, 1], [2, 1, 2], [1, 2, 1]])
    branch3 = np.array([[1, 2, 1], [2, 1, 2], [1, 2, 2]])
    branch4 = np.array([[2, 1, 2], [1, 1, 2], [2, 1, 2]])
    branch5 = np.array([[1, 2, 2], [2, 1, 2], [1, 2, 1]])
    branch6 = np.array([[2, 2, 2], [1, 1, 1], [2, 1, 2]])
    branch7 = np.array([[2, 2, 1], [2, 1, 2], [1, 2, 1]])
    branch8 = np.array([[2, 1, 2], [2, 1, 1], [2, 1, 2]])
    branch9 = np.array([[1, 2, 1], [2, 1, 2], [2, 2, 1]])
    br1 = mh.morph.hitmiss(skel, branch1)
    br2 = mh.morph.hitmiss(skel, branch2)
    br3 = mh.morph.hitmiss(skel, branch3)
    br4 = mh.morph.hitmiss(skel, branch4)
    br5 = mh.morph.hitmiss(skel, branch5)
    br6 = mh.morph.hitmiss(skel, branch6)
    br7 = mh.morph.hitmiss(skel, branch7)
    br8 = mh.morph.hitmiss(skel, branch8)
    br9 = mh.morph.hitmiss(skel, branch9)
    return br1 + br2 + br3 + br4 + br5 + br6 + br7 + br8 + br9


def endPoints(skel):
    endpoint1 = np.array([[0, 0, 0],
                          [0, 1, 0],
                          [2, 1, 2]])

    endpoint2 = np.array([[0, 0, 0],
                          [0, 1, 2],
                          [0, 2, 1]])

    endpoint3 = np.array([[0, 0, 2],
                          [0, 1, 1],
                          [0, 0, 2]])

    endpoint4 = np.array([[0, 2, 1],
                          [0, 1, 2],
                          [0, 0, 0]])

    endpoint5 = np.array([[2, 1, 2],
                          [0, 1, 0],
                          [0, 0, 0]])

    endpoint6 = np.array([[1, 2, 0],
                          [2, 1, 0],
                          [0, 0, 0]])

    endpoint7 = np.array([[2, 0, 0],
                          [1, 1, 0],
                          [2, 0, 0]])

    endpoint8 = np.array([[0, 0, 0],
                          [2, 1, 0],
                          [1, 2, 0]])

    ep1 = mh.morph.hitmiss(skel, endpoint1)
    ep2 = mh.morph.hitmiss(skel, endpoint2)
    ep3 = mh.morph.hitmiss(skel, endpoint3)
    ep4 = mh.morph.hitmiss(skel, endpoint4)
    ep5 = mh.morph.hitmiss(skel, endpoint5)
    ep6 = mh.morph.hitmiss(skel, endpoint6)
    ep7 = mh.morph.hitmiss(skel, endpoint7)
    ep8 = mh.morph.hitmiss(skel, endpoint8)
    ep = ep1 + ep2 + ep3 + ep4 + ep5 + ep6 + ep7 + ep8
    return ep


def pruning(skeleton, size):
    for i in range(0, size):
        endpoints = endPoints(skeleton)
        endpoints = np.logical_not(endpoints)
        skeleton = np.logical_and(skeleton, endpoints)
    return skeleton

def neighbours(x,y,image):
    """Return 8-neighbours of image point P1(x,y), in a clockwise order"""
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1;
    # return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1], img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]
    return [ img[x_1][y], img[x_1][y_1], img[x][y_1], img[x1][y_1], img[x1][y], img[x1][y1], img[x][y1], img[x_1][y1] ]


def getSkeletonIntersection(skeleton):
    """ Given a skeletonised image, it will give the coordinates of the intersections of the skeleton.

    Keyword arguments:
    skeleton -- the skeletonised image to detect the intersections of

    Returns:
    List of 2-tuples (x,y) containing the intersection coordinates
    """
    # A biiiiiig list of valid intersections             2 3 4
    # These are in the format shown to the right         1 C 5
    #                                                    8 7 6
    validIntersection = [[0, 1, 0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0, 0, 1], [1, 0, 0, 1, 0, 1, 0, 0],
                         [0, 1, 0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 0, 0, 1, 0],
                         [0, 1, 0, 0, 1, 0, 0, 1], [1, 0, 1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0, 1],
                         [0, 1, 0, 1, 0, 0, 0, 1], [0, 1, 0, 1, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1, 0, 1],
                         [1, 0, 1, 0, 0, 0, 1, 0], [1, 0, 1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0, 1, 0],
                         [1, 0, 0, 0, 1, 0, 1, 0], [1, 0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 0, 0, 1, 1, 1],
                         [1, 1, 0, 0, 1, 0, 0, 1], [0, 1, 1, 1, 0, 0, 1, 0], [1, 0, 1, 1, 0, 0, 1, 0],
                         [1, 0, 1, 0, 0, 1, 1, 0], [1, 0, 1, 1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 0, 1, 1],
                         [1, 1, 0, 1, 1, 0, 1, 0], [1, 1, 0, 0, 1, 0, 1, 0], [0, 1, 1, 0, 1, 0, 1, 0],
                         [0, 0, 1, 0, 1, 0, 1, 1], [1, 0, 0, 1, 1, 0, 1, 0], [1, 0, 1, 0, 1, 1, 0, 1],
                         [1, 0, 1, 0, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0, 0, 1], [0, 1, 0, 0, 1, 0, 1, 1],
                         [0, 1, 1, 0, 1, 0, 0, 1], [1, 1, 0, 1, 0, 0, 1, 0], [0, 1, 0, 1, 1, 0, 1, 0],
                         [0, 0, 1, 0, 1, 1, 0, 1], [1, 0, 1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 0, 1, 1, 0],
                         [1, 0, 1, 1, 0, 1, 0, 0]];
    image = skeleton.copy();
    # image = image/255;
    intersections = list();
    for x in range(1, len(image) - 1):
        for y in range(1, len(image[x]) - 1):
            # If we have a white pixel
            if image[x][y] == 1:
                neighbours1 = neighbours(x, y, image);
                valid = True;
                if neighbours1 in validIntersection:
                    intersections.append((y, x));
    # Filter intersections to make sure we don't count them twice or ones that are very close together
    for point1 in intersections:
        for point2 in intersections:
            if (((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) < 10 ** 2) and (point1 != point2):
                intersections.remove(point2);
    # Remove duplicates
    intersections = list(set(intersections));
    return intersections


def removeCross(cross, distanceMap, thinned):
    width = distanceMap[cross[1]][cross[0]]

    width = width.astype('uint8')

    for x in range(0, width * 2 + 1):
        for y in range(0, width * 2 + 1):
            # print(x - width, y - width)

            i = cross[1] - width + x
            j = cross[0] - width + y

            if i>=0 and i<thinned.shape[0] and j>=0 and j<thinned.shape[1]:
                thinned[i][j] = False

    return thinned


def get_mean_diameter(diameter_count):
    """
    Promediar el diámetro:
    Recibe un diccionario con los diámetros detectados y y la cantidad respectiva.
    Retorna el promedio del diámetro
    """
    suma = 0
    contador = 0
    for k,v in diameter_count.items():
        suma += k*v
        contador += v
    return (suma / contador)

def removeCrosses(skeleton, distance_map, crosses):
    """ Eliminar intersecciones """
    uncrossed = skeleton.copy()
    # recorrer puntos de interseccion:
    for cross in crosses:
        x = cross[0]
        y = cross[1]
        width = math.ceil(distance_map[y,x])
        x_start = x - width if x - width >= 0 else 0
        y_start = y - width if y - width >= 0 else 0
        # por cada punto eliminar su alrededor segun valor del mapa de distancia
        uncrossed[y_start:y+width+1,x_start:x+width+1] = False

    return uncrossed


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
    p = p+s
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