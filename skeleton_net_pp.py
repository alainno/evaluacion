from skimage import morphology
from skimage.measure import regionprops
from skimage import morphology
import numpy as np
import cv2
from postprocessing_utils import PostProcess

def skeleton_net_pp(sample_prediction):
    diametros = {}

    # distance_map = model(sample_img)
    distance_map = cv2.imread(sample_prediction, 0)
    dm_normalized = cv2.normalize(distance_map, None, 0, 255, cv2.NORM_MINMAX)
    segmentos = PostProcess(dm_normalized, 40, 16)
    regiones = regionprops(segmentos)

    distance_map = (distance_map / 255) / 0.01

    # recorrer todas las regiones detectadas:
    for i,region in enumerate(regiones):
        # obtenemos el segmento actual:
        segmento = (segmentos==i+1)
        # obtenemos el skeleton del segmento:
        seg_skeleton = morphology.skeletonize(segmento, method='lee')
        # obtenemos sus diametros desde el mapa de distancia:
        seg_diametros = np.floor(distance_map[seg_skeleton>0]*2)
        # contamos los diametros
        unique, counts = np.unique(seg_diametros, return_counts=True)
        seg_diametros_count = dict(zip(unique, counts))
        for k,v in seg_diametros_count.items():
            diametros[k] = diametros.get(k,0) + v

    return diametros