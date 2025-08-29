import cv2
import yaml
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
from argparse import Namespace
from preprocessing.utils import get_central_crop
from tracking.features import get_patch_features, build_heatmap
from interaction.coordinate import get_tracking_feature


def run_tracking(config):
    onnx_session = ort.InferenceSession(config.onnx_model_path)
    tracking_feature = get_tracking_feature(config, onnx_session)

    cap = cv2.VideoCapture(config.path_to_input_video)

    out_writer = cv2.VideoWriter(
        config.path_to_output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        cap.get(cv2.CAP_PROP_FPS),
        (config.img_size, config.img_size * 2)
    )
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=total_frames, desc="Processing video") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_cropped = get_central_crop(frame, config.img_size)
            patch_features = get_patch_features(frame_cropped, onnx_session)  # [196, 384]
            
            heatmap_resized = build_heatmap(patch_features, tracking_feature, config)
            
            # this thing is not necessary and added for nicer vizualization
            heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (
                heatmap_resized.max() - heatmap_resized.min() + 1e-6
            )

            heatmap_norm = 1 - heatmap_norm

            heatmap_colored = cv2.applyColorMap(
                (heatmap_norm*255).astype(np.uint8),
                cv2.COLORMAP_JET
            )

            final_image = np.vstack((frame_cropped, 
                                     cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR)))
            out_writer.write(final_image)

            pbar.update(1)

    cap.release()
    out_writer.release()


if __name__ == "__main__":
    with open("config.yaml", 'r') as file:
        config = Namespace(**yaml.safe_load(file))
    run_tracking(config)