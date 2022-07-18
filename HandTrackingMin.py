import cv2
import time
import mediapipe as mp


cap = cv2.VideoCapture(0)

mphands = mp.solutions.hands
hands = mphands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    ret, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)
    # extract hands from results
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for hlm in results.multi_hand_landmarks:
            for id, lm in enumerate(hlm.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)

                if id == 4:
                    cv2.circle(frame, (cx,cy), 20, (0,255,0), cv2.FILLED)


            mpDraw.draw_landmarks(frame, hlm, mphands.HAND_CONNECTIONS)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)



    cv2.imshow('webcame', frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
