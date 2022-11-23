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
                cx,cy, cz = int(lm.x*w), int(lm.y*h), int(lm.z)
                lmlist.append([id,cx,cy,cz])
            if draw:
                cv2.circle(image,(cx,cy), 15 , (255,0,255), cv2.FILLED)
        return lmlist

    def calc_bounding_rect(self,image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_array = np.empty((0, 2), int)
        for _, landmark in enumerate(landmarks):
            landmark_x = int(landmark[1])
            landmark_y = int(landmark[2])
            landmark_point = [np.array((landmark_x, landmark_y))]
            landmark_array = np.append(landmark_array, landmark_point, axis=0)
        x, y, w, h = cv2.boundingRect(landmark_array)
        return [x, y, w, h]

    def draw_bounding_rect(self,use_brect, image, brect):
        if use_brect:
            # Outer rectangle
            cv2.rectangle(image, (brect[0], brect[1]), (brect[2] + brect[0], brect[3]+ brect[1]),
                        (199, 12, 34), 1)

        return image

    def draw_info(self,image, fps):
        cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)
        return image
    def calc_landmark_list(image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        # Keypoint
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            # landmark_z = landmark.z

            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args

def main():
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    cap = cv2.VideoCapture(0)
    tracker = hand_tracker()
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    print(volume.GetVolumeRange()) 

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

        success, image = cap.read()
        image = cv2.flip(image,1)
        image = tracker.hands_finder(image)
        debug_image = copy.deepcopy(image)
        lmList = tracker.position_finder(image)
        image.flags.writeable = False
        #results = hands.process(image)
        image.flags.writeable = True
        if len(lmList) != 0:
            #thumb and index detection and output & setting the volume
            thumb_x, thumb_y = lmList[4][1], lmList[4][2]
            index_x, index_y = lmList[8][1], lmList[8][2]
            length = int(math.hypot(index_x-thumb_x, index_y-thumb_y))
            
            
            #drawing
            brect = tracker.calc_bounding_rect(debug_image, lmList)
            debug_image = tracker.draw_bounding_rect(True, debug_image, brect)
            perc = int((length/brect[3]) * 100)
            print(" - x : ",brect[0] ," - y : ", brect[1]," - w : ", brect[2], " - h : ",brect[3]," - length : ",length,
            " - percentage : ", perc)
            cv2.circle(debug_image, (thumb_x, thumb_y), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(debug_image, (index_x, index_y), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(debug_image, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 255), 3)

            volumeValue = np.interp(perc, [7, 85], [-65.25, 0.0])
            volume.SetMasterVolumeLevel(volumeValue, None)

        else:
            print("No hands Detected")
        debug_image = tracker.draw_info(debug_image, int(fps))
        cv2.imshow('Hand-Tracker volume control',debug_image)
        key = cv2.waitKey(1)
        if key == 27:
            break