import numpy as np
import msgpack as mp
import msgpack_numpy as mpn
import cv2
from cv2 import aruco


def data_unpacker(data_path):
    with open(data_path, 'rb') as f:
        unpacker = mp.Unpacker(f, object_hook=mpn.decode)
    return unpacker

