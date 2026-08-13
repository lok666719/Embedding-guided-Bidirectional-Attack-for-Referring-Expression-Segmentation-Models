import cv2
import numpy as np
import torch

def load_image(img_path):
    image_np = cv2.imread(img_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    return image_np

def show_pic(image, name="pic"):
    import matplotlib.pyplot as plt
    plt.axis('off')
    plt.imshow(image)
    plt.title(name, fontsize=32)
    plt.show()

clean_path = '/public/chenxingbai/chenxingbai/EVF-SAM-main/assets/zebra.jpg'
adv_path = '/public/chenxingbai/chenxingbai/EVF-SAM-main/adv_example/adv_bus.png'

clean_image = load_image(clean_path)
adv_image = load_image(adv_path)

show_pic(clean_image, 'clean')
show_pic(adv_image, 'adv')
pertub = adv_image - clean_image
show_pic(pertub, 'pertub')