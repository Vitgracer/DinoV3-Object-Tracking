import cv2
from tracking.features import get_patch_features
from preprocessing.utils import get_central_crop
import matplotlib.pyplot as plt


def get_initial_frame(path_to_input_video, img_size):
    cap = cv2.VideoCapture(path_to_input_video)
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("failed")
    cap.release()

    initial_frame_resized = get_central_crop(first_frame, img_size)
    return initial_frame_resized

def get_tracking_feature(config, onnx_session):
    initial_frame = get_initial_frame(config.path_to_input_video, config.img_size)
    num_patches_per_dim = config.img_size // config.patch_size

    tracking_feature = None

    # here we process the first click, 
    # where user choses a region to track
    def onclick(event):
        nonlocal tracking_feature
        cx, cy = int(event.xdata), int(event.ydata)
        patch_x = cx // config.patch_size
        patch_y = cy // config.patch_size
        patch_idx_ref = patch_y * num_patches_per_dim + patch_x
        patch_features = get_patch_features(initial_frame, onnx_session)
        tracking_feature = patch_features[patch_idx_ref]
        plt.close() 

    # here we show an initial image
    fig, ax = plt.subplots(num="Initial image")
    ax.imshow(cv2.cvtColor(initial_frame, cv2.COLOR_BGR2RGB))
    ax.set_title("CLICK THE OBJECT TO TRACK", fontsize=16, color="red")
    ax.axis("off")
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    return tracking_feature
