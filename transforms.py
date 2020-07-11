from PIL import ImageFilter

def blur_gauss(img):
    return img.filter(ImageFilter.GaussianBlur(3))