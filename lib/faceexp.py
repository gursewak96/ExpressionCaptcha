import math
import cv2
from scipy.spatial import distance as dist
import numpy as np
from collections import OrderedDict


EAR_THRESH = 0.20  #Eye aspect ratio threshold
MAR_THRESH = 0.4   #Mouth aspect ration threshold

COUNTER_BLINK = 0
COUNTER_YAWN = 0

EYE_FRAMES = 0
MAR_FRAMES = 0

FACIAL_LANDMARKS_IDXS = OrderedDict([
                        ("mouth",(48,68)),
                        ("outer_mouth",(48,61)),
                        ("inner_mouth",(60,68)),
                        ("right_eyebrow",(17,22)),
                        ("left_eyebrow",(22,27)),
                        ("right_eye",(36,42)),
                        ("left_eye",(42,48)),
                        ("nose",(27,36)),
                        ("jaw",(0,17))
                        ])

def eye_aspect_ratio(eye):
    """
    Takes the list of eye
    returns the aspect ratio of the given eye
    """
    A = dist.euclidean(eye[1],eye[5])
    B = dist.euclidean(eye[2],eye[4])
    C = dist.euclidean(eye[0],eye[3])
    return (A+B)/(2.0*C)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[3],mouth[9])
    B = dist.euclidean(mouth[0],mouth[6])
    return A/B

def shape_to_np(shape, dtype="int"):
    """
    Takes the shape dlib: full_object_detection and dtype
    Return the array of coordinates of landmarks
    """
    #initialise the list of x-y coordinates
    coord = np.zeros((shape.num_parts,2),dtype=dtype)

    #loop over facial landmarks and convert them into x,y tupeles
    for i in range(0,shape.num_parts):
        coord[i] = (shape.part(i).x,shape.part(i).y)

    return coord
def getEar(shape,isLeft = True):
    """
    Assume the dlib: object
    Returns the mouth aspect ratio and eye aspect Ratio
    """
    shape = shape_to_np(shape)
    if isLeft:
        (leftEyeStart,leftEyeEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
        leftEye = shape[leftEyeStart:leftEyeEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        return leftEAR
    else:
        (rightEyeStart, rightEyeEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
        rightEye = shape[rightEyeStart:rightEyeEnd]
        rightEar = eye_aspect_ratio(rightEye)
        return rightEar

def getMar(shape):
    shape = shape_to_np(shape)
    (mStart,mEnd) = FACIAL_LANDMARKS_IDXS["outer_mouth"]
    mouth = shape[mStart:mEnd]

    return mouth_aspect_ratio(mouth)
