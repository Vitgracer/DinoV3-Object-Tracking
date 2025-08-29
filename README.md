![GitHub last commit](https://img.shields.io/github/last-commit/Vitgracer/DinoV3-Object-Tracking?color=blue)
![GitHub repo size](https://img.shields.io/github/repo-size/Vitgracer/DinoV3-Object-Tracking?color=green)
![GitHub stars](https://img.shields.io/github/stars/Vitgracer/DinoV3-Object-Tracking?style=social)
![GitHub forks](https://img.shields.io/github/forks/Vitgracer/DinoV3-Object-Tracking?style=social)
![Python](https://img.shields.io/badge/Python-3776AB.svg?logo=python&logoColor=white)

# 🎉 DINOv3 Object Tracking Demo
⚠️ **It is not an official Meta product.**  

This project shows how to track objects in videos using the powerful [**DINOv3**](https://github.com/facebookresearch/dinov3) model. Let's dive in! 🏊‍♂️  

---

## 🦖 What is DINOv3?

[**DINOv3**](https://github.com/facebookresearch/dinov3) is a self-supervised vision transformer (ViT) model created by Meta.  
It can:

- Understand images without needing labeled data
- Produce super-robust feature embeddings for image patches  
- Be used for image segmentation, object tracking, zero-shot classification, and more 
- Works even if the object rotates, scales, or changes appearance   

🤓 In short: *DinoV3 just knows. Everything. Period.*

---

## 💪 What does this project do?

This project is a **fun demo of object tracking** on videos using [**DINOv3**](https://github.com/facebookresearch/dinov3). 

How it works:

1. Take the **first frame** of your video.  
2. **Click on the object** you want to track using your mouse. 
3. Pass the frame through [**DINOv3**](https://github.com/facebookresearch/dinov3), which splits the image into **patches**. Each patch gets its own **feature vector**.  We are interested in the feature vector of the user' selected patch.
4. Compute the **cosine similarity** between the feature vector of the selected patch and all other patches of other frames.  
5. Use these similarities to create a similarity **heatmap**. More 🟠 "orange" - more similar! 

---

## 🏃‍♂️ How to install and run

### Step 1: Create a virtual environment
```bash
python -m venv dino-venv
source dino-venv/Scripts/activate 
```

### Step 2: Install dependencies
```bash
pip install onnxruntime
pip install opencv-python
pip install tqdm matplotlib
```

### Step 3: Download the model

- Go to [HuggingFace ONNX community](https://huggingface.co/onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX-MHA-scores/tree/main) and download a DINOv3 model.
- Place it in the *model/* folder.
- We used **fp16 ViT-S**, but you can try any other variant

### Step 4: Config is "all you need" 😅
- Open *config.py* and set *path_to_input_video* to your video file
- The video will be cropped to a square and resized to 224×224 for convenience

### Step 5: RUN! 
```bash
python run.py
```
- The first frame will appear
- Click on the object you want to track
- The script will process all frames and save a tracked video 🎥

## 🔑 Licenses

- Code in this repository is licensed under the [MIT License](./LICENSE).
- The DINOv3 model weights are licensed under the [DINOv3 License](./LICENSE-DINOv3) by Meta.
- Weights were downloaded from HuggingFace [ONNX community](https://huggingface.co/onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX-MHA-scores/tree/main).
- By using the model weights, you agree to the terms of the DINOv3 License.
- Images/Videos used in this project are sourced from [Pixabay](https://pixabay.com/) and [Unsplash](https://unsplash.com/) under their respective licenses.