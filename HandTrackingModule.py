import cv2
import time
import mediapipe as mp


class HandDetector():
    def __init__(self, mode=False, maxHands=2,modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.modelComplexity = modelComplexity
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionCon, self.trackCon)

        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]



    def findHands(self, frame, draw=True):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frameRGB)
        # extract hands from results
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for hlm in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, hlm,
                                               self.mpHands.HAND_CONNECTIONS)
        return frame

    def findPosition(self, frame, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])

                if draw:
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
        return self.lmList

    def fingersUp(self):
        fingers = []
        #for checking whether thumb is up or not if it is up it is sending 1
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] -1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):

            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:

                fingers.append(1)
            else:
                fingers.append(0)
        return fingers




def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        ret, frame1 = cap.read()
        frame1 = detector.findHands(frame1, draw=True)
        lmList = detector.findPosition(frame1, draw=True)
        if len(lmList) != 0:
            print(lmList[4])


        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(frame1, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cv2.imshow('webcame', frame1)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
