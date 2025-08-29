import cv2
import numpy as np


def get_central_crop(img, img_size):
    h, w = img.shape[:2]
    side = min(h, w)

    top = (h - side) // 2
    left = (w - side) // 2

    cropped = img[top:top+side, left:left+side]
    cropped = cv2.resize(cropped, (img_size, img_size), 
                         interpolation=cv2.INTER_CUBIC)
    return cropped

def preprocess_frame(frame):
    """ 
    Here we expect to see BGR 224x224 frame,
    and preprocess it for model input.
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_data = frame.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_data = (img_data - mean) / std
    img_data = np.transpose(img_data, (2,0,1))
    img_data = np.expand_dims(img_data, 0)
    return img_data