import argparse
import json
import matplotlib.pyplot as plt
import os
import sys
import tqdm
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoTokenizer, BitsAndBytesConfig
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (AverageMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)
from evf_sam_attacker import evf_sam_attacker
from evf_sam2_attacker import evf_sam2_attacker
import pickle

def stretch_to_16_9(image):
    """
    Stretch the input image to 16:9 aspect ratio by scaling both axes.
    This method does not crop or pad, but may distort the image.

    Parameters:
        image (np.ndarray): Input image in H x W x C format.

    Returns:
        np.ndarray: Resized image with 16:9 aspect ratio.
    """
    h, w = image.shape[:2]
    original_ratio = w / h
    target_ratio = 16 / 9

    # 以长边为基准，缩放另一边，使变形后的图像是 16:9
    if original_ratio > target_ratio:
        # 图像过宽，压缩宽度
        new_w = int(h * target_ratio)
        new_h = h
    else:
        # 图像过高，拉伸高度
        new_w = w
        new_h = int(w / target_ratio)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized

def get_adv_image_path(ori_image_pth, adv_examples_dir):
    filename = os.path.basename(ori_image_pth)
    filename_base, ext = os.path.splitext(filename)
    adv_filename = filename_base + '_adv' + ext
    new_path = os.path.join(adv_examples_dir, adv_filename)
    return new_path


def resize_to_16_9_by_long_edge(image, padding_color=(0, 0, 0)):
    """
    Resize a numpy image to 16:9 ratio by adjusting the short side,
    while keeping the long side unchanged.

    Parameters:
        image (np.ndarray): Input image in H x W x C format.
        padding_color (tuple): RGB color used for padding if needed.

    Returns:
        np.ndarray: Resized image with 16:9 aspect ratio.
    """
    h, w = image.shape[:2]
    target_ratio = 16 / 9

    if w / h > target_ratio:
        # 宽是长边，按宽度保持不变，计算目标高度
        new_h = int(w / target_ratio)
        if new_h <= h:
            # 裁剪高度
            offset = (h - new_h) // 2
            resized = image[offset:offset + new_h, :, :]
        else:
            # 高度不够，填充
            pad_total = new_h - h
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            resized = cv2.copyMakeBorder(image, pad_top, pad_bottom, 0, 0,
                                         cv2.BORDER_CONSTANT, value=padding_color)
    else:
        # 高是长边，按高度保持不变，计算目标宽度
        new_w = int(h * target_ratio)
        if new_w <= w:
            # 裁剪宽度
            offset = (w - new_w) // 2
            resized = image[:, offset:offset + new_w, :]
        else:
            # 宽度不够，填充
            pad_total = new_w - w
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            resized = cv2.copyMakeBorder(image, 0, 0, pad_left, pad_right,
                                         cv2.BORDER_CONSTANT, value=padding_color)

    return resized

import cv2
import numpy as np

def highlight_full_mask_with_contour(image, mask, alpha=0.45, contour_color=(255, 0, 255), contour_thickness=2):
    """
    保留所有mask细节，增强可视化效果（背景变暗 + 分割区域加轮廓线）

    image: 原图 (H, W, 3), uint8
    mask: 分割掩码 (H, W), bool 或 0/1 array
    """
    # Step 1: 对 mask 做 resize 和 bool 化
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = mask.astype(bool)

    # Step 2: 背景变暗
    dimmed_img = (image * alpha).astype(np.uint8)

    # Step 3: 恢复掩码区域的亮度
    result_img = dimmed_img.copy()
    result_img[mask] = image[mask]

    # Step 4: 添加所有mask区域的边界轮廓（包括碎片区域）
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    result_img = cv2.drawContours(result_img, contours, -1, contour_color, contour_thickness)

    return result_img


def caculate_aver_loss(total_list):
    averge_loss_list = []
    for i in range(100):
        total = 0
        for j in range(750):
            total += total_list[j][i]
        avverage_loss_per_iter = total / 750
        averge_loss_list.append(avverage_loss_per_iter)
    return averge_loss_list

def get_adv_save_dir(image_path, adv_dir, attackname):
    file_name = os.path.basename(image_path)
    name, ext = os.path.splitext(file_name)
    new_file_name = f"{name}_adv{ext}"
    save_dir =os.path.join(adv_dir, attackname)
    save_dir = os.path.join(save_dir, new_file_name)
    return save_dir

def get_adv_save_dir1(image_path, adv_dir):
    file_name = os.path.basename(image_path)
    name, ext = os.path.splitext(file_name)
    new_file_name = f"{name}_adv{ext}"
    save_dir =os.path.join(adv_dir, new_file_name)
    return save_dir

def show_pic(image, name="pic"):
    import matplotlib.pyplot as plt
    plt.axis('off')
    plt.imshow(image)
    plt.title(name, fontsize=32)
    plt.show()

def IoU(mask1, mask2=None):
    if mask1.__class__ == torch.Tensor:
        mask1 = mask1.detach().cpu().numpy()
    if mask2 is None:
        mask2 = np.ones_like(mask1, dtype=bool)
    elif mask2.__class__ == torch.Tensor:
        mask2 = mask2.detach().cpu().numpy()
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return 100 * np.sum(intersection) / np.sum(union)

def save_image(image):
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("output.jpg", img_bgr)

def load_np_image(image_path):
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    return image_np

evf_sam2_checkpoint = '/public/chenxingbai/chenxingbai/EVF-SAM-main/check points/evf-sam2'
evf_sam_checkpoint = '/public/chenxingbai/chenxingbai/EVF-SAM-main/check points/evf-sam'
train_set_path = '/public/chenxingbai/chenxingbai/EVF-SAM-main/splited_dataset_new/reccoco/testA/train.p'
test_set_path = '/public/chenxingbai/chenxingbai/EVF-SAM-main/splited_dataset_new/reccoco/testA/test.p'
# /public/chenxingbai/chenxingbai/EVF-SAM-main/splited_dataset_new/refcoco+/train.p
# /public/chenxingbai/chenxingbai/EVF-SAM-main/splited_dataset_new/refcoco+/test.p
# /public/chenxingbai/chenxingbai/EVF-SAM-main/splited_dataset_new/refcocog/all_>=5/train.p
# /public/chenxingbai/chenxingbai/EVF-SAM-main/splited_dataset_new/refcocog/all_>=5/test.p
torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
tokenizer = AutoTokenizer.from_pretrained(
    evf_sam_checkpoint,
    padding_side="right",
    use_fast=False,
)
torch_dtype = torch.float16
kwargs = {"torch_dtype": torch_dtype}
sam1_model = evf_sam_attacker.from_pretrained(evf_sam_checkpoint, low_cpu_mem_usage=True, **kwargs)
sam1_model.cuda()
sam1_model.eval()
# sam2_model =evf_sam2_attacker.from_pretrained(evf_sam2_checkpoint, low_cpu_mem_usage=True, **kwargs)
# sam2_model.cuda()
# sam2_model.eval()
train_set = pickle.load(open(train_set_path, "rb"))
test_set = pickle.load(open(test_set_path, "rb"))
# example_dir = '/public/chenxingbai/chenxingbai/EVF-SAM-main/sam1_adv_examples_recoco+/ours v1'
example_dir = '/public/chenxingbai/chenxingbai/EVF-SAM-main/sam1_adv_examples(iter:100,loss:-10)recoco_testA/ours v1 0.01'
imge_list = [266240,221187,464917,444445,522288]
if __name__ == "__main__":
    for image in test_set['images']:
        image_id = image['id']
        # if image_id not in imge_list:
        #     continue
        if image_id != 522288:
            continue
        # image = train_set['images'][i]
        image_path = image['file_name']
        adv_image_path = get_adv_image_path(image_path, example_dir)
        # prompt = train_set['img2refs'][image_id][0]['sentences'][0]['sent']
        adv_cv2_img = load_np_image(adv_image_path)
        clean_cv2_img = load_np_image(image_path)
        prompt_list = []
        for refs in test_set['img2refs'][image_id]:
            for ref in refs['sentences']:
                prompt_list.append(ref['sent'])
        # prompt_list = ['motorcycle']
        # test_mask = generate_concentrated_bool_array(adv_cv2_img.shape[:2], 5000)
        # adv_cv2_img[test_mask] = (
        #         adv_cv2_img * 0.5
        #         + test_mask[:, :, None].astype(np.uint8) * np.array([50, 120, 220]) * 0.5
        #     )[test_mask]
        # show_pic(adv_cv2_img)
        clean = True
        if clean == True:
            with torch.no_grad():
                adv_mask = sam1_model.test(clean_cv2_img, prompt_list, tokenizer)[0]
        else:
            with torch.no_grad():
                adv_mask = sam1_model.test(adv_cv2_img, prompt_list, tokenizer)[0]
        pred_mask = adv_mask.detach().cpu().numpy()
        pred_mask = pred_mask > 0
        img_list = []
        for i in range(pred_mask.shape[0]):
            if clean == True:
                current_img = highlight_full_mask_with_contour(clean_cv2_img, pred_mask[i])
                img_list.append(current_img)
            else:
                current_img = highlight_full_mask_with_contour(adv_cv2_img, pred_mask[i])
                img_list.append(current_img)
        for img in img_list:
            show_pic(img)

        torch.cuda.empty_cache()