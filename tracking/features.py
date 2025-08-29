import cv2
import numpy as np
from preprocessing.utils import preprocess_frame


def get_patch_features(frame, onnx_session):
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    
    img_data = preprocess_frame(frame)
    features = onnx_session.run([output_name], {input_name: img_data})[0]
    
    # 0-index token os cls token (global iamge edscription)
    # [1-4] tokens - register tokens (for global context)
    patch_features = features[:, 5:, :][0] 

    # transform feature vector to the vector with the length 1 for cosine distance
    patch_features_norm = patch_features / np.linalg.norm(patch_features, axis=1, keepdims=True)
    
    # [196, 384]
    return patch_features_norm

def build_heatmap(patch_features, ref_feature, config):
    """ 
    Cosine distance = (x * y) / (len(x) * len(y)), 
    but we have vectors with the length = 1.

    Big cosine distance - features are similar. 
    """
    num_patches_per_dim = config.img_size // config.patch_size
    cos_sim = patch_features @ ref_feature  # [196]
    heatmap = cos_sim.reshape(num_patches_per_dim, num_patches_per_dim)
    heatmap_resized = cv2.resize(heatmap, (config.img_size, config.img_size), 
                                 interpolation=cv2.INTER_CUBIC)
    return heatmap_resized