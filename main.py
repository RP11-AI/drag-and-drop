# +--------------------------------------------------------------------------------------------------------------------+
# |                                                                                                  (module) main.py ||
# |                                                                                             Author:Pauliv, Rômulo ||
# |                                                                                      Copyright 2022, RP11.AI Ltd. ||
# |                                                                                           https://keepo.io/rp11ai ||
# |                                                                                                   Encoding: UTF-8 ||
# +--------------------------------------------------------------------------------------------------------------------+

# imports -------------------------------------------------------------------------------------------------------------|
import cv2
import mediapipe as mp
from typing import Union


# ---------------------------------------------------------------------------------------------------------------------|


def resolution_cap(dev: cv2.VideoCapture) -> dict[str, int]:
    """
    Takes a frame from the capture of the cv2.VideoCapture function to get the resolution data.
    :param dev: cv2.VideoCapture function with the device or video parameter to be used.
    :return: Returns a dictionary containing height and width.
    """
    _, frame = dev.read()
    return {'width': frame.shape[1], 'height': frame.shape[0]}


class Drag(object):
    def __init__(self, directory: Union[str, int] = 0) -> None:
        self.cap = cv2.VideoCapture(directory)
        self.resolution_cap = resolution_cap(self.cap)
        self.mp_hands = mp.solutions.hands
        self.image, self.results, self.middle_finger_tip = None, None, {}

    def Frame(self) -> bool:
        """
        Reads the function capture and plays it in a np.array.
        :return: Returns whether the capture of the frame was successful or not.
        """
        auth, self.image = self.cap.read()
        return auth

    def Hands(self, engine: mp.python.solutions.hands.Hands) -> None:
        """
        MediaPipe Hands handling and detection process.
        :param engine: Hand detection engine.
        """
        self.image.flags.writeable = False
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.results = engine.process(self.image)
        self.image.flags.writeable = True
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

    def Landmarks(self) -> None:
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.middle_finger_tip['x'] = int(hand_landmarks.landmark[
                                                      self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP
                                                  ].x * self.resolution_cap['width'])
                self.middle_finger_tip['y'] = int(hand_landmarks.landmark[
                                                      self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP
                                                  ].y * self.resolution_cap['height'])

                cv2.circle(self.image, center=(self.middle_finger_tip['x'], self.middle_finger_tip['y']),
                           radius=5, color=(255, 0, 255), thickness=1)

    def Show(self) -> None:
        """
        Show the video.
        """
        cv2.imshow('Hands', cv2.flip(self.image, 1))

    def DestroyCap(self) -> None:
        """
        Release the capture or video device and destroys all windows generated by opencv.
        """
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    system = Drag()
    with system.mp_hands.Hands(model_complexity=0,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as hands:
        while system.cap.isOpened():
            success = system.Frame()
            if not success:
                continue
            system.Hands(engine=hands)
            system.Landmarks()
            system.Show()
            if cv2.waitKey(5) & 0xFF == 27:
                break
    system.DestroyCap()
