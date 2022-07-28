# +--------------------------------------------------------------------------------------------------------------------+
# |                                                                                             (module) drag_drop.py ||
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


def mean_xy(coord: list[dict[str, int]]) -> tuple[int, int]:
    """
    having two points (x', y') and (x'', y'') the returned function (x, y) which will be the midpoint of the line.
    :param coord: points (x', y') and (x'', y'')
    :return: returns the midpoint (x, y)
    """
    param, xy_minor = ('x', 'y'), {}
    for i in param:
        xy_minor[i] = min(coord[0][i], coord[1][i])
    return (int(abs((coord[0]['x'] - coord[1]['x']) / 2) + xy_minor['x']),
            int(abs((coord[0]['y'] - coord[1]['y']) / 2) + xy_minor['y']))


def rectangle_center(center: tuple[int, int], radius: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Having the coordinate of the center of the rectangle (x, y) it will return the upper left coordinate (x', y') and
    the lower right coordinate (x'', y'') of the rectangle.
    :param center: (x, y) rectangle center.
    :param radius: Proportional distance from the center to the edges of the rectangle, generating a square.
    :return: Returns (x', y') and (x'', y'') referring to the upper left and lower right points of the rectangle.
    """
    pt1 = (center[0] - radius, center[1] - radius)
    pt2 = (center[0] + radius, center[1] + radius)
    return pt1, pt2


class Drag(object):
    def __init__(self, directory: Union[str, int] = 0, record: bool = False) -> None:
        self.cap = cv2.VideoCapture(directory)
        self.resolution_cap = resolution_cap(self.cap)
        self.mp_hands = mp.solutions.hands
        self.image, self.results = None, None
        self.finger_1, self.finger_2, self.fingers = {}, {}, [{}, {}]
        self.distance, self.button = None, None
        self.initial_position = rectangle_center((100, 100), 50)
        if record:
            self.output_video = cv2.VideoWriter("my_video.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                                (self.resolution_cap['width'], self.resolution_cap['height']))

    def Frame(self) -> bool:
        """
        Reads the function capture and plays it in a np.array.
        :return: Returns whether the capture of the frame was successful or not.
        """
        auth, self.image = self.cap.read()
        self.finger_1, self.finger_2 = {}, {}
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

    def Landmarks(self, finger_1: int = 8, finger_2: int = 4) -> list[dict[str, int]]:
        """
        Find the landmarks of each finger according to the following list:
            0. WRIST
            1, 2, 3, 4. THUMB (CMC, MCP, IP, TIP)
            5, 6, 7, 8. INDEX FINGER (MCP, PIP, DIP, TIP)
            9, 10, 11, 12. MIDDLE FINGER (MCP, PIP, DIP, TIP)
            13, 14, 15, 16. RING FINGER (MCP, PIP, DIP, TIP)
            17, 18, 19, 20. PINKY (MCP, PIP, DIP, TIP)
        :param finger_1: Landmark number.
        :param finger_2: Landmark number.
        :return: A list containing a dictionary [x, y] with the coordinates of the two given fingers.
        """
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                for num_id, landmark in enumerate(hand_landmarks.landmark):
                    if num_id == finger_1:
                        self.finger_1['x'] = int(landmark.x * self.resolution_cap['width'])
                        self.finger_1['y'] = int(landmark.y * self.resolution_cap['height'])
                    if num_id == finger_2:
                        self.finger_2['x'] = int(landmark.x * self.resolution_cap['width'])
                        self.finger_2['y'] = int(landmark.y * self.resolution_cap['height'])
        self.fingers = [self.finger_1, self.finger_2]
        return self.fingers

    def Distance(self) -> None:
        """
        Distance between the fingers.
        """
        if self.results.multi_hand_landmarks:
            self.distance = round(((self.finger_1['x'] - self.finger_2['x']) ** 2 +
                                   (self.finger_1['y'] - self.finger_2['y']) ** 2) ** (1 / 2), 2)

    def DistanceActivation(self, min_distance: int = 50) -> bool:
        """
        Activates a Boolean return when the distance between the specified fingers decreases.
        :param min_distance: < than the value, the return will be true.
        :return: bool.
        """
        if self.results.multi_hand_landmarks:
            if self.distance < min_distance:
                self.button = True
                cv2.circle(img=self.image, center=mean_xy(self.fingers), radius=8, color=(0, 180, 0),
                           thickness=cv2.FILLED)
            else:
                self.button = False
        return self.button

    def Object(self) -> None:
        color_pf = (255, 255, 0)
        if self.results.multi_hand_landmarks:
            if self.button:
                if self.initial_position[0][0] <= mean_xy(self.fingers)[0] <= self.initial_position[1][0] and \
                        self.initial_position[0][1] <= mean_xy(self.fingers)[1] <= self.initial_position[1][1]:
                    self.initial_position = rectangle_center(center=mean_xy(self.fingers), radius=50)
                    color_pf = (60, 255, 255)

        overlay = self.image.copy()
        cv2.rectangle(img=overlay, pt1=self.initial_position[0], pt2=self.initial_position[1],
                      color=color_pf, thickness=-1)
        cv2.addWeighted(src1=overlay, alpha=0.5, src2=self.image, beta=0.5, gamma=0, dst=self.image)

    def ShowLandmarks(self, coord: bool = True, obj_circle: bool = True) -> None:
        """
        Show landmarks in video or capture.
        :param obj_circle: Circles showing the position of the fingers.
        :param coord: Displays the coordinate in the image
        """
        if self.results.multi_hand_landmarks:
            for i in self.fingers:
                if obj_circle:
                    cv2.circle(img=self.image, center=(i['x'], i['y']), radius=3, color=(255, 255, 255),
                               thickness=cv2.FILLED)
                if coord:
                    cv2.putText(img=self.image, text='({a}, {b})'.format(a=i['x'], b=i['y']), org=(i['x'], i['y'] + 5),
                                fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=0.4, color=(255, 255, 255),
                                thickness=1)

    def ShowDistance(self, coord: bool = True, obj_line: bool = True) -> None:
        """
        Shows the distance between the fingers.
        :param obj_line: Line showing distance between fingers.
        :param coord: displays the coordinate in the image.
        """
        if self.results.multi_hand_landmarks:
            if obj_line:
                cv2.line(self.image, pt1=(self.finger_1['x'], self.finger_1['y']),
                         pt2=(self.finger_2['x'], self.finger_2['y']), color=(255, 0, 255), thickness=1)
            if coord:
                cv2.putText(img=self.image, text=str(self.distance), org=mean_xy(self.fingers),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=0.4, color=(255, 255, 255), thickness=1)

    def Show(self) -> None:
        """
        Show the video.
        """
        cv2.imshow('Hands', cv2.flip(self.image, 1))

    def Record(self) -> None:
        """
        Record frames.
        """
        self.output_video.write(self.image)

    def DestroyCap(self) -> None:
        """
        Release the capture or video device and destroys all windows generated by opencv.
        """
        self.cap.release()
        cv2.destroyAllWindows()
