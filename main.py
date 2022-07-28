# +--------------------------------------------------------------------------------------------------------------------+
# |                                                                                                           main.py ||
# |                                                                                             Author:Pauliv, Rômulo ||
# |                                                                                      Copyright 2022, RP11.AI Ltd. ||
# |                                                                                           https://keepo.io/rp11ai ||
# |                                                                                                   Encoding: UTF-8 ||
# +--------------------------------------------------------------------------------------------------------------------+

# imports -------------------------------------------------------------------------------------------------------------|
import drag_drop as dd
import cv2
# ---------------------------------------------------------------------------------------------------------------------|

if __name__ == '__main__':
    system = dd.Drag(record=False)
    with system.mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5,
                               min_tracking_confidence=0.5, max_num_hands=1) as hands:
        while system.cap.isOpened():
            success = system.Frame()
            if not success:
                continue
            system.Hands(engine=hands)

            system.Landmarks()
            system.Distance()
            system.DistanceActivation(min_distance=50)
            system.Object()

            system.ShowLandmarks(coord=True, obj_circle=True)
            system.ShowDistance(coord=True, obj_line=True)

            system.Show()
            if cv2.waitKey(5) & 0xFF == 27:
                break
    system.DestroyCap()
