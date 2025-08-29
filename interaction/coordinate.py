import cv2
from preprocessing.utils import get_central_crop


def get_initial_frame(path_to_input_video, img_size):
    cap = cv2.VideoCapture(path_to_input_video)
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("failed")
    cap.release()

    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    first_frame_rgb_cropped = get_central_crop(first_frame_rgb)
    first_frame_rgb_resized = cv2.resize(first_frame_rgb_cropped, (img_size, img_size))
    
    return first_frame_rgb_resized

def get_tracking_coordinate(config):
    initial_frame = get_initial_frame(config.path_to_input_video, config.img_size)
    return initial_frame, None