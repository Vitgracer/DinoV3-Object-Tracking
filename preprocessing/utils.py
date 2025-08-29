
def get_central_crop(img):
    h, w = img.shape[:2]
    side = min(h, w)

    top = (h - side) // 2
    left = (w - side) // 2

    cropped = img[top:top+side, left:left+side]
    return cropped