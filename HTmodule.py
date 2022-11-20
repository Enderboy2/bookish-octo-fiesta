import mediapipe as mp
import cv2

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
def main():
    cap = cv2.VideoCapture(0)
    tracker = hand_tracker()
    while True:
        success, image = cap.read()
        image = tracker.hands_finder(image,True)
        lmList = tracker.position_finder(image,0,True)
        if len(lmList) != 0:
            print(lmList[4])
        cv2.imshow("Video",image)
        cv2.waitKey(1)