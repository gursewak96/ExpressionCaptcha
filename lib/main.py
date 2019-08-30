import cv2
import dlib
import argparse
import random
import time
import clock
import numpy as np
import common
import faceexp
import orient

ap = argparse.ArgumentParser()
ap.add_argument("-c","--camera",required=False,help="Camera number from where to get feed.",default=0)
args = vars(ap.parse_args())
print(args)

font = cv2.FONT_HERSHEY_SIMPLEX
#create a list of gestures
gestures = ["Blink Eye","Yawn","Look Left","Look Right","Look Up","Look Down","Roll Left","Roll Right"]
gesture = -1

CaptchaSolved = 0
CaptchaMissed = 0
TotalCaptcha = 0
isFailed = False
isPassed = False

counter = 0

cv2.namedWindow("Gesture")
cv2.moveWindow("Gesture",50,50)
cv2.namedWindow("Live Feed")
cv2.moveWindow("Live Feed",370,50)

#create the frame and initialise the camera
cap = cv2.VideoCapture(int(args['camera']))
_,frame = cap.read()
orient.setParam(frame)

while True:
    menu = np.zeros((412,312,3), np.uint8)
    #if passed then print it for 5 sec
    if isPassed :
        cv2.putText(menu,"Passed",(10,90),font,1,(0,255,0),2)
        cv2.putText(menu,"Getting New Captcha in 5 seconds:"+" "+str(counter),(10,110),font,0.5,(0,255,0),1)
        if counter > 5 :
            gesture = -1
            isPassed = not isPassed


    elif isFailed :
        cv2.putText(menu,"Failed",(10,90),font,1,(0,00,255),2)
        cv2.putText(menu,"Getting New Captcha in 5 seconds:"+" "+str(counter),(10,110),font,0.5,(0,255,0),1)
        if counter > 5 :
            gesture = -1
            isFailed = not isFailed
    #if counter is -1 then get the new gesture and set counter to zero
    elif  gesture == -1:
        gesture = int(random.randint(0,len(gestures)-1))
        counter = clock.restart()

    _,frame = cap.read()


    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    (face,rect) = common.detect_face(gray)
    if (not isFailed) and (not isPassed):

        if(face != None):
            shape = common.predictor(gray,face)

            #for eyeblinking
            if gesture == 0:

                ear = faceexp.getEar(shape)

                if ear < faceexp.EAR_THRESH:
                    faceexp.EYE_FRAMES += 1
                elif faceexp.EYE_FRAMES > 2:
                    faceexp.COUNTER_BLINK += 1
                    faceexp.EYE_FRAMES = 0
                if faceexp.COUNTER_BLINK > 0:
                    isPassed = True
                    CaptchaSolved += 1
                    faceexp.COUNTER_BLINK = 0
                    faceexp.EYE_FRAMES = 0
                    counter = clock.restart()
                elif counter > 5:
                    isFailed = True
                    CaptchaMissed += 1
                    counter = clock.restart()


            if gesture == 1:
                mar = faceexp.getMar(shape)

                if mar >= faceexp.MAR_THRESH:
                    faceexp.MAR_FRAMES += 1
                else:
                    if faceexp.MAR_FRAMES > 6:
                        faceexp.COUNTER_YAWN += 1
                    faceexp.MAR_FRAMES = 0
                if faceexp.COUNTER_YAWN > 0:
                    isPassed = True
                    faceexp.COUNTER_YAWN = 0
                    faceexp.MAR_FRAMES = 0
                    counter = clock.restart()
                elif counter > 5:
                    isFailed = True
                    counter = clock.restart()

            if gesture == 2:
                imagePointsVector = orient.getImagePoints(common.shape_to_np(shape))
                rotation_vector,translation_vector = orient.getRotAndTransVector(imagePointsVector)
                rmat, _ = cv2.Rodrigues(rotation_vector)
                angles = orient.rotationMatrixToEulerAngles(rmat)
                pitch, yaw, roll = (angles[0]*57.230-360)%360,angles[1]*57.230,angles[2]*57.230
                if yaw < -20:
                    isPassed = True
                    counter = clock.restart()
                elif counter > 5:
                    isFailed = True
                    counter = clock.restart()

            if gesture == 3:
                imagePointsVector = orient.getImagePoints(common.shape_to_np(shape))
                rotation_vector,translation_vector = orient.getRotAndTransVector(imagePointsVector)
                rmat, _ = cv2.Rodrigues(rotation_vector)
                angles = orient.rotationMatrixToEulerAngles(rmat)
                pitch, yaw, roll = (angles[0]*57.230-360)%360,angles[1]*57.230,angles[2]*57.230
                if yaw > 40:
                    isPassed = True
                    counter = clock.restart()
                elif counter > 5:
                    isFailed = True
                    counter = clock.restart()

            if gesture == 4:
                imagePointsVector = orient.getImagePoints(common.shape_to_np(shape))
                rotation_vector,translation_vector = orient.getRotAndTransVector(imagePointsVector)
                rmat, _ = cv2.Rodrigues(rotation_vector)
                angles = orient.rotationMatrixToEulerAngles(rmat)
                pitch, yaw, roll = (angles[0]*57.230-360)%360,angles[1]*57.230,angles[2]*57.230
                if pitch <= 163:
                    isPassed = True
                    counter = clock.restart()
                elif counter > 5:
                    isFailed = True
                    counter = clock.restart()

            if gesture == 5:
                imagePointsVector = orient.getImagePoints(common.shape_to_np(shape))
                rotation_vector,translation_vector = orient.getRotAndTransVector(imagePointsVector)
                rmat, _ = cv2.Rodrigues(rotation_vector)
                angles = orient.rotationMatrixToEulerAngles(rmat)
                pitch, yaw, roll = (angles[0]*57.230-360)%360,angles[1]*57.230,angles[2]*57.230
                if pitch > 185:
                    isPassed = True
                    counter = clock.restart()
                elif counter > 5:
                    isFailed = True
                    counter = clock.restart()

            if gesture == 6:
                imagePointsVector = orient.getImagePoints(common.shape_to_np(shape))
                rotation_vector,translation_vector = orient.getRotAndTransVector(imagePointsVector)
                rmat, _ = cv2.Rodrigues(rotation_vector)
                angles = orient.rotationMatrixToEulerAngles(rmat)
                pitch, yaw, roll = (angles[0]*57.230-360)%360,angles[1]*57.230,angles[2]*57.230
                if roll > 20:
                    isPassed = True
                    counter = clock.restart()
                elif counter > 5:
                    isFailed = True
                    counter = clock.restart()

            if gesture == 7:
                imagePointsVector = orient.getImagePoints(common.shape_to_np(shape))
                rotation_vector,translation_vector = orient.getRotAndTransVector(imagePointsVector)
                rmat, _ = cv2.Rodrigues(rotation_vector)
                angles = orient.rotationMatrixToEulerAngles(rmat)
                pitch, yaw, roll = (angles[0]*57.230-360)%360,angles[1]*57.230,angles[2]*57.230
                if roll < -20:
                    isPassed = True
                    counter = clock.restart()
                elif counter > 5:
                    isFailed = True
                    counter = clock.restart()


            cv2.putText(menu,gestures[gesture],(10,90),font,0.8,(255,255,0))
            cv2.putText(menu,"Solve within 5 seconds: "+str(counter),(10,110),font,0.5,(255,255,255))

            cv2.rectangle(frame,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0,255,255),3)
    #make sure if counter is incrementing on every loop make sure to get new gesture if counter
    #counter exceeds 5
    counter = clock.getCount()
    #show the frame
    cv2.imshow("Gesture",menu)
    cv2.imshow("Live Feed",frame)

    #wait for the key
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
