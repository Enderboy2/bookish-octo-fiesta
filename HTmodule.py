import mediapipe as mp
import cv2
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

class hand_tracker:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5,modelComplexity=1,trackCon=0.5):
        self.mode = mode
        self.max_hands = maxHands
        self.detection_con = detectionCon
        self.model_complex = modelComplexity
        self.track_con = trackCon
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands,self.model_complex,self.detection_con, self.track_con)
        self.mp_draw = mp.solutions.drawing_utils
    def hands_finder(self,image,draw=True):
        image_RGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image_RGB)
        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(image, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return image
    def position_finder(self,image, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h,w,c = image.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                lmlist.append([id,cx,cy])
            if draw:
                cv2.circle(image,(cx,cy), 15 , (255,0,255), cv2.FILLED)

        return lmlist
    def distance_calculate(self, p1, p2):
        """p1 and p2 in format (thumb_x,thumb_y) and (index_x,index_y) tuples"""
        self.dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
        return self.dis
def main():
    cap = cv2.VideoCapture(0)
    tracker = hand_tracker()
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    print(volume.GetVolumeRange()) 
    while True:
        success, image = cap.read()
        image = cv2.flip(image,1)
        image = tracker.hands_finder(image)
        lmList = tracker.position_finder(image)

        if len(lmList) != 0:
            thumb_x, thumb_y = lmList[4][1], lmList[4][2]
            index_x, index_y = lmList[8][1], lmList[8][2]
            length = math.hypot(index_x-thumb_x, index_y-thumb_y)
            print(length)
            volumeValue = np.interp(length, [20, 250], [-65.25, 0.0])
            volume.SetMasterVolumeLevel(volumeValue, None)
            
            cv2.circle(image, (thumb_x, thumb_y), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(image, (index_x, index_y), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(image, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 255), 3)
        else:
            print("No hands Detected")

        cv2.imshow('Hand-Tracker volume control',image)
        key = cv2.waitKey(1)
        if key == 27:
            break
        