from utils import getSkeletonIntersection, removeCrosses
import numpy as np
import math
import cv2
from skimage import morphology

def new_distance_transform(sample_img):
    """ Implementación del algoritmo de transformación de distancia de Ziabari, 2008 """
    # Declaramos la variable de salida:
    diametros = {}
    # Binarización de la micrografia:
    rgb_img = cv2.imread(sample_img)
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    _,bin_img = cv2.threshold(gray_img,127,255,cv2.THRESH_BINARY)
    # Exqueletizamos la imagen binaria:
    skeleton = morphology.skeletonize(bin_img, method='lee')  
    # Obtenemos su mapa de distancia:
    distance_map = cv2.distanceTransform(bin_img, cv2.DIST_C, cv2.DIST_MASK_3)
    # Obtenemos las intersecciones del skeleton:
    crosses = getSkeletonIntersection(skeleton)
    # Removemos las intersecciones segun el skeleton y el mapa de distancia:
    uncrossed = removeCrosses(skeleton, distance_map, crosses)
    # computamos los diametros con el mapa de distancia y el skeleton sin intersecciones:
    diametros_dm = np.floor(distance_map[uncrossed>0]*2)
    unique, counts = np.unique(diametros_dm, return_counts=True)
    diametros = dict(zip(unique, counts))

    return diametros
