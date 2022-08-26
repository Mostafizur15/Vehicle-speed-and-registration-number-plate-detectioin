import cv2
import dlib
import time
from datetime import datetime
import numpy as np
import pytesseract
import imutils
pytesseract.pytesseract.tesseract_cmd = "C:\Program Files\Tesseract-OCR/tesseract.exe"

cascade= cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")

carCascade = cv2.CascadeClassifier('two_wheeler.xml')
carCascade1 = cv2.CascadeClassifier('vech.xml')
video = cv2.VideoCapture('Traffic.mp4')

WIDTH = 1280
HEIGHT = 720  

fps = 1

ini_tracker = {}
end_tracker = {} 


def contour(image):
    triangle_cnt = np.array( [[1280,0], [920,0],[1280,360]] )
    triangle_cnt2 = np.array( [[0,0],[360,0],[0,300]] )

    cv2.drawContours(image, [triangle_cnt], 0, (0,0,0), -1)
    cv2.drawContours(image, [triangle_cnt2], 0, (0,0,0), -1)

    return image

def estimateSpeed(carID,dis):
    timeDiff = int(end_tracker[carID]-ini_tracker[carID])
    speed = round(dis/timeDiff*fps*3.6,2)
    return speed


def find(id,chk):
     print(id)
     gaussian_blur=cv2.GaussianBlur(chk,(7,7),2)
     for i in range(id,id+1):
        v1=i+.5
        v2=-1*(v1-1)
        print(v1,v2)
        sharp3 = cv2.addWeighted(chk,v1,gaussian_blur,v2,0)
        gray=cv2.cvtColor(sharp3,cv2.COLOR_BGR2GRAY)
        nplate=cascade.detectMultiScale(gray,1.1,4)
        print(nplate)
        return nplate
     


def saveCar(speed,img):
    now = datetime.today().now()
    nameCurTime = now.strftime("%d-%m-%Y-%H-%M-%S-%f")

    link = 'overspeeding/Detail_info/'+nameCurTime+'.jpeg'
    link1 = 'overspeeding/Number_plate/'+nameCurTime+'.jpeg'

    scale=3.5
    width=int(img.shape[1]*scale)
    height=int(img.shape[0]*scale)
    dimension=(width,height)
    tmpimg=cv2.resize(img,dimension,interpolation=cv2.INTER_AREA)
    median=cv2.medianBlur(tmpimg,5)
    img=median
    #gaussian_blur=cv2.GaussianBlur(img,(7,7),3)
    #sharp3 = cv2.addWeighted(img,5.5,gaussian_blur,-4.5,0)
    #img=sharp3
    #cv2.imwrite(link,img)


    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    nplate=cascade.detectMultiScale(gray,1.1,4)

    f=0
    for i in range(1,10):

        for (x,y,w,h) in nplate:
            f=1
            wT,hT,cT=img.shape
            a,b=(int(0.02*wT),int(0.02*hT))
            plate=img[y+a:y+h-a,x+b:x+w-b]
            cv2.imshow("plate",plate)
        if(f==0):
            nplate=find(i,img)
        else:
            cv2.imwrite(link,img)
            cv2.imwrite(link1,plate)
        if(f==1):
            break


def trackMultipleObjects():
    rectangleColor = (0, 255, 0)
    frameCounter = 0
    currentCarID = 0
    carTracker = {}
    velocity=[None]*1000
    
    while True:
        rc, image = video.read()
        if type(image) == type(None):
            break

        frameTime = time.time()
        image = cv2.resize(image, (WIDTH, HEIGHT))[240:720,0:1280]
        resultImage = contour(image)
        resultImage = image
        cv2.line(resultImage,(0,120),(1280,120),(0,0,255),2)
        cv2.line(resultImage,(0,360),(1280,360),(0,0,255),2)

        #cv2.line(resultImage,(0,270),(1280,270),(255,0,0),2)
        #cv2.line(resultImage,(0,30),(1280,30),(255,0,0),2)

        frameCounter = frameCounter + 1

        carIDtoDelete = []

        for carID in carTracker.keys():
            trackingQuality = carTracker[carID].update(image)

            if trackingQuality < 7:
                carIDtoDelete.append(carID)

        for carID in carIDtoDelete:
            carTracker.pop(carID, None)

        if (frameCounter%60 == 0):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cars = carCascade.detectMultiScale(gray, 1.01, 1,0,(1,1))
            cars1 = carCascade1.detectMultiScale(gray, 1.1, 13, 0, (24, 24))

            
            
            for (_x, _y, _w, _h) in cars1:
                x = int(_x)
                y = int(_y)
                wdth = int(_w)
                hght = int(_h)

                xmid = x + 0.5*wdth
                ymid = y + 0.5*hght

                matchCarID = None

                for carID in carTracker.keys():
                    trackedPosition = carTracker[carID].get_position()

                    tx = int(trackedPosition.left())
                    ty = int(trackedPosition.top())
                    twdth = int(trackedPosition.width())
                    thght = int(trackedPosition.height())

                    txmid = tx + 0.5 * twdth
                    tymid = ty + 0.5 * thght

                    if ((tx <= xmid <= (tx + twdth)) and (ty <= ymid <= (ty + thght)) and (x <= txmid <= (x + wdth)) and (y <= tymid <= (y + hght))):
                        matchCarID = carID


                if matchCarID is None:
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(image, dlib.rectangle(x, y, x + wdth, y + hght))

                    carTracker[currentCarID] = tracker

                    currentCarID = currentCarID + 1

        for carID in carTracker.keys():
            trackedPosition = carTracker[carID].get_position()

            tx = int(trackedPosition.left())
            ty = int(trackedPosition.top())
            tw = int(trackedPosition.width())
            th = int(trackedPosition.height())

            cv2.rectangle(resultImage, (tx, ty), (tx + tw, ty + th), rectangleColor, 2)
            cv2.putText(resultImage,"Id="+ str(carID) , (tx,ty-10), cv2.FONT_HERSHEY_DUPLEX,.7, (0, 255, 0), 1)

            

            if carID not in ini_tracker and 360 > ty+th > 120 and ty<120:
                ini_tracker[carID] = frameTime
            elif carID in ini_tracker and carID not in end_tracker and 360 < ty+th:
                end_tracker[carID] = frameTime
                df=end_tracker[carID]-ini_tracker[carID]
                velocity[carID] = estimateSpeed(carID,45)
                if velocity[carID] > 25:
                    print('CAR-ID : {} : {} kmph - OVERSPEED Time: {}'.format(carID, velocity[carID],df))
                    saveCar(velocity[carID],image[ty:ty+th, tx:tx+tw])
                else:
                    print('CAR-ID : {} : {} kmph time: {}'.format(carID, velocity[carID],df))
                #v=speed
            if(velocity[carID]!=None):
                cv2.putText(resultImage,"V = "+ str(velocity[carID])+"km/h", (tx,ty+th+20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)

            
        cv2.imshow('result', resultImage)
        if cv2.waitKey(33) == 27:
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    trackMultipleObjects()