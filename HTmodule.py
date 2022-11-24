import mediapipe as mp
import cv2
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import copy
import argparse
import time

class hand_tracker:
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5,modelComplexity=1,trackCon=0.5):
        self.mode = mode
        self.max_hands = maxHands
        self.min_detection_confidence = detectionCon
        self.min_tracking_confidence = trackCon
        self.mp_hands = mp.solutions.hands
        self.model_complex = modelComplexity
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands,self.model_complex,self.min_detection_confidence, self.min_tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils

    def hands_finder(self,image,draw=True):

        """Draw the joints(landmarks) and their connection in the hand

        :param: image - img
        :param: draw - bool - wheter to draw or not
        :return: image - img - image with drawn hands

        """

        image_RGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image_RGB)
        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(image, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return image

    def position_finder(self,image, handNo=0, draw=True):

        """Find the landmarks on the image

        :param: image
        :param: handNo - int - number of hands
        :param: draw - bool - wheter to draw or not
        :return: lmlist - array - list of positions of the landmarks

        """

        lmlist = []
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h,w,c = image.shape
                cx,cy, cz = int(lm.x*w), int(lm.y*h), int(lm.z)
                lmlist.append([id,cx,cy,cz])
            if draw:
                cv2.circle(image,(cx,cy), 15 , (255,0,255), cv2.FILLED)
        return lmlist

    def calc_bounding_rect(self,landmarks):

        """Calculate the coordinates of the outer rectangle

        :params: landmarks - array - landmarks list
        :return: [x, y, w, h]  - array - x, y, width and height of the rectangle

        """

        landmark_array = np.empty((0, 2), int)
        for _, landmark in enumerate(landmarks):
            landmark_x = int(landmark[1])
            landmark_y = int(landmark[2])
            landmark_point = [np.array((landmark_x, landmark_y))]
            landmark_array = np.append(landmark_array, landmark_point, axis=0)
        x, y, w, h = cv2.boundingRect(landmark_array)
        return [x, y, w, h]

    def draw_bounding_rect(self,use_brect, image, brect):

        """Draw the bounding rect

        :param: use_brect - bool - whether to draw or not
        :param: image - img - image to be drawn on
        :param: brect - array - x, y, width and height of the rectangle
        :return: image - img - image with drawn bounding rectangle

        """

        if use_brect:
            cv2.rectangle(image, (brect[0], brect[1]), (brect[2] + brect[0], brect[3]+ brect[1]),
                        (199, 12, 34), 1)
        return image

    def draw_info(self,image, fps, perc):

        """Draw the info on the image
        
        :param: image - img - image to be drawn on
        :param: fps - int
        :param: perc - int
        :return: image - img - image with drawn info

        """

        cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, "%" + str(perc), (0,185), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)
        return image

def main():
    # Loading models
    cap = cv2.VideoCapture(0)
    tracker = hand_tracker()
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    print(volume.GetVolumeRange()) 
    
    # Fps
    fps_start_time = 0
    fps = 0

    while True:

        #fps init
        fps_end_time = time.time()
        time_diff = fps_end_time - fps_start_time
        fps = 1/(time_diff)
        fps_start_time = fps_end_time

        #break if pressed ESC
        key = cv2.waitKey(10)
        if key == 27:  # ESC
            break

        #init image
        _, image = cap.read()
        image = cv2.flip(image,1)
        image = tracker.hands_finder(image)
        lmList = tracker.position_finder(image)
        perc = 0
        if len(lmList) != 0:

            #Caculating distance bet. thumb <-> index & getting the boundin rectangle
            thumb_x, thumb_y = lmList[4][1], lmList[4][2]
            index_x, index_y = lmList[8][1], lmList[8][2]
            length = int(math.hypot(index_x-thumb_x, index_y-thumb_y))
            brect = tracker.calc_bounding_rect(lmList)
            image = tracker.draw_bounding_rect(True, image, brect)
            d = math.hypot(brect[3], brect[2])
            perc_d = int( (length/d) * 100)
            perc_h = int( (length/brect[3]) * 100)
            print(d,perc_d ," - ",perc_h)
            perc = perc_d
            #Drawing
            cv2.circle(image, (thumb_x, thumb_y), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(image, (index_x, index_y), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(image, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 255), 3)

            #Setting the volume
            volumeValue = np.interp(perc_d, [12, 60], [-65.25, 0.0])
            volume.SetMasterVolumeLevel(volumeValue, None)

        else:
            print("No hands Detected")

        image = tracker.draw_info(image, int(fps), perc)
        cv2.imshow('Hand-Tracker volume control',image) #Showing the window