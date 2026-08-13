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
from skimage.metrics import structural_similarity
from evf_sam_attacker import evf_sam_attacker
import pickle



def get_adv_image_path(ori_image_pth, adv_examples_dir):
    filename = os.path.basename(ori_image_pth)
    filename_base, ext = os.path.splitext(filename)
    adv_filename = filename_base + '_adv' + ext
    new_path = os.path.join(adv_examples_dir, adv_filename)
    return new_path




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

def normalize_to_float01(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to float32 in the range [0, 1].

    Supports common uint8 images and floating-point images whose
    pixel values are in either [0, 1] or [0, 255].
    """
    image = image.astype(np.float32)

    if image.max() > 1.0:
        image = image / 255.0

    return np.clip(image, 0.0, 1.0)


def resize_adv_to_clean(
    adv_image: np.ndarray,
    clean_image: np.ndarray,
) -> np.ndarray:
    """
    Resize the adversarial image to the spatial size of the clean image.
    """
    clean_h, clean_w = clean_image.shape[:2]
    adv_h, adv_w = adv_image.shape[:2]

    if (adv_h, adv_w) == (clean_h, clean_w):
        return adv_image

    # INTER_AREA is generally suitable for downsampling;
    # INTER_LINEAR is used when at least one dimension is enlarged.
    if adv_h >= clean_h and adv_w >= clean_w:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR

    resized_adv = cv2.resize(
        adv_image,
        dsize=(clean_w, clean_h),  # OpenCV uses (width, height)
        interpolation=interpolation,
    )

    return resized_adv


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
example_dir = '/public/chenxingbai/chenxingbai/EVF-SAM-main/sam1_adv_examples(iter:100,loss:-10)recoco_testA/single_p'
path_list = ['/public/chenxingbai/chenxingbai/EVF-SAM-main/sam1_adv_examples(iter:100,loss:-10)/attack_sam_k',
             '/public/chenxingbai/chenxingbai/EVF-SAM-main/sam1_adv_examples(iter:100,loss:-10)/SRA',
             '/public/chenxingbai/chenxingbai/EVF-SAM-main/sam1_adv_examples(iter:100,loss:-10)recoco_testA/coattack',
             '/public/chenxingbai/chenxingbai/EVF-SAM-main/sam1_adv_examples(iter:100,loss:-10)recoco_testA/single_p',
             '/public/chenxingbai/chenxingbai/EVF-SAM-main/sam1_adv_examples(iter:100,loss:-10)recoco_testA/multi_p',
             '/public/chenxingbai/chenxingbai/EVF-SAM-main/sam1_adv_examples(iter:100,loss:-10)recoco_testA/segpgd_single_p',
             '/public/chenxingbai/chenxingbai/EVF-SAM-main/sam1_adv_examples(iter:100,loss:-10)recoco_testA/segpgd_multi_p',
             '/public/chenxingbai/chenxingbai/EVF-SAM-main/sam1_adv_examples(iter:100,loss:-10)recoco_testA/ours v1 0.01']
imge_list = [266240,221187,464917,444445,522288]
if __name__ == "__main__":
    for i in range(len(test_set['images'])):
    # for image in tqdm.tqdm(train_set['images']):
        if i >= 200:
            break
        image = test_set['images'][i]
    # for image in test_set['images']:
        image_id = image['id']

        # if image_id not in imge_list:
        #     continue
        # if image_id != 522288:
        #     continue
        # image = train_set['images'][i]
        image_path = image['file_name']
        adv_image_path = get_adv_image_path(image_path, example_dir)
        # prompt = train_set['img2refs'][image_id][0]['sentences'][0]['sent']
        adv_cv2_img = load_np_image(adv_image_path)
        clean_cv2_img = load_np_image(image_path)
        if adv_cv2_img is None:
            raise FileNotFoundError(
                f"Failed to load adversarial image: {adv_image_path}"
            )

        if clean_cv2_img is None:
            raise FileNotFoundError(
                f"Failed to load clean image: {image_path}"
            )

        if adv_cv2_img.ndim != 3 or adv_cv2_img.shape[2] != 3:
            raise ValueError(
                f"Adversarial image {image_id} has invalid shape: "
                f"{adv_cv2_img.shape}"
            )

        if clean_cv2_img.ndim != 3 or clean_cv2_img.shape[2] != 3:
            raise ValueError(
                f"Clean image {image_id} has invalid shape: "
                f"{clean_cv2_img.shape}"
            )

        # 1. Resize adversarial image to the clean-image resolution.
        adv_cv2_img = resize_adv_to_clean(
            adv_image=adv_cv2_img,
            clean_image=clean_cv2_img,
        )
