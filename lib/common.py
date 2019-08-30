import cv2
import dlib
import numpy as np
from collections import OrderedDict

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shapepredictor/shape_predictor_68_face_landmarks.dat")

def shape_to_np(shape, dtype="int"):
    """
    Takes the shape dlib: full_object_detection and dtype
    Return the array of coordinates of landmarks
    """
    coord = np.zeros((shape.num_parts,2),dtype = dtype)

    for i in range(0,shape.num_parts):
        coord[i] = (shape.part(i).x,shape.part(i).y)

    return coord

def detect_face(grayImage):
    """
    return the face and the rectangle
    """

    #to get the closest faces
    x = 0
    y = 0
    w = 0
    h = 0
    max_area = 0

    faces = detector(grayImage)
    face = None
    rect = (x,y,w,h)

    for _face in faces:
        _x = _face.left()
        _y = _face.top()
        _w = _face.right() - _x
        _h = _face.bottom() - _y
        if _w*_h > max_area:
            max_area = _w*_h
            x = _x
            y = _y
            w = _w
            h = _h
            face = _face
            rect = x,y,w,h

    if face == None:
        return None,None

    else:
        return face,rect
