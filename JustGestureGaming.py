import time
import cv2
import mediapipe as mp
from pynput.keyboard import Key, Controller
import math

class PoseDetector:
    def __init__(self):
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=False,
                                     model_complexity=0,
                                     smooth_landmarks=True,
                                     enable_segmentation=False,
                                     smooth_segmentation=True,
                                     min_detection_confidence=0.5,
                                     min_tracking_confidence=0.5)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        return self.lmList

    def findDistance(self, p1, p2, img=None, color=(255, 0, 255), scale=5):
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)

        if img is not None:
            cv2.circle(img, (x1, y1), 10, color, cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, color, cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), color, max(1, scale // 3))
            cv2.circle(img, (cx, cy), 10, color, cv2.FILLED)

        return length, info, img

    def findAngle(self, img, p1, p2, p3, draw=True):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 5, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 5, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x3, y3), 5, (255, 0, 0), cv2.FILLED)
            cv2.putText(img, str(int(angle)), (x2 - 20, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        return angle

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = PoseDetector()
    keyboard = Controller()
    counter = 0

    while True:
        success, img = cap.read()
        img = cv2.resize(img, (750, 550))
        img = cv2.flip(img, 1)
        detector.findPose(img)
        lmList = detector.getPosition(img)

        if lmList:
            p1, p2 = lmList[1][1:], lmList[23][1:]
            left, right = lmList[18][1:], lmList[19][1:]
            l, _, _ = detector.findDistance(p1, p2)
            l1, _, _ = detector.findDistance(left, right)

            if l1 < 80:
                if counter == 0:
                    keyboard.press(Key.up)
                    keyboard.release(Key.up)
            if p1[1] > 250:
                if counter == 0:
                    keyboard.press(Key.down)
                    keyboard.release(Key.down)
            if left[0] < 150:
                keyboard.press(Key.left)
                keyboard.release(Key.left)
            if right[0] > 600:
                keyboard.press(Key.right)
                keyboard.release(Key.right)

            counter += 1

        if counter == 11:
            counter = 0

        cv2.imshow("Temple Run", img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()
