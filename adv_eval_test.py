import argparse
import json
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
def record_iou_ratio(current_log, output_list, gt_list):
    for i in range(output_list.shape[0]):
        iou_background = IoU(mask1=output_list[i])
        iou_foreground = IoU(mask1=output_list[i], mask2=gt_list[i])
        output_num = int(torch.sum(output_list[i]))
        gt_num = np.sum(gt_list[i])
        ratio = output_num / gt_num
        if ratio < 0.3:
            is_success = True
        else:
            is_success = False
        current_log['iou_foreground_list'].append(iou_foreground)
        current_log['iou_background_list'].append(iou_background)
        current_log['is_success'].append(is_success)

def to_bool_mask(segmentation, ori_shape, adv_shape):
    segmentation = np.array(segmentation).reshape((-1, 2))
    mask = np.zeros(ori_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [segmentation.astype(np.int32)], 1)
    h, w = adv_shape
    resized_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    resized_mask = resized_mask > 0
    return resized_mask

def get_adv_image_path(ori_image_pth, adv_examples_dir):
    filename = os.path.basename(ori_image_pth)
    filename_base, ext = os.path.splitext(filename)
    adv_filename = filename_base + '_adv' + ext
    new_path = os.path.join(adv_examples_dir, adv_filename)
    return new_path

def get_adv_save_dir(image_path, adv_dir, attackname):
    file_name = os.path.basename(image_path)
    name, ext = os.path.splitext(file_name)
    new_file_name = f"{name}_adv{ext}"
    save_dir =os.path.join(adv_dir, attackname)
    save_dir = os.path.join(save_dir, new_file_name)
    return save_dir
def show_pic(image, name="pic"):
    import matplotlib.pyplot as plt
    plt.axis('off')
    plt.imshow(image)
    plt.title(name, fontsize=32)
    plt.show()

# def IoU(mask1, mask2=None):
#     if mask1.__class__ == torch.Tensor:
#         mask1 = mask1.detach().cpu().numpy()
#     if mask2 is None:
#         mask2 = np.ones_like(mask1, dtype=bool)
#     elif mask2.__class__ == torch.Tensor:
#         mask2 = mask2.detach().cpu().numpy()
#     intersection = np.logical_and(~mask1, mask2)
#     union = np.logical_or(~mask1, mask2)
#     iou = 100 * np.sum(intersection) / np.sum(union)
#     return round(iou, 2)

def IoU(mask1, mask2=None):
    if mask1.__class__ == torch.Tensor:
        mask1 = mask1.detach().cpu().numpy()
    background_flag = False
    if mask2 is None:
        mask2 = np.ones_like(mask1, dtype=bool)
        background_flag = True
    elif mask2.__class__ == torch.Tensor:
        mask2 = mask2.detach().cpu().numpy()
    if background_flag == True:
        intersection = np.logical_and(~mask1, mask2)
        union = np.logical_or(~mask1, mask2)
    else:
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
    iou = 100 * np.sum(intersection) / np.sum(union)
    return round(iou, 2)

def load_np_image(image_path):
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    return image_np

evf_sam2_checkpoint = './check points/evf-sam2'
evf_sam_checkpoint = './check points/evf-sam'
train_set_path = './splited_dataset_new/reccoco/testA/train.p'
test_set_path = './splited_dataset_new/reccoco/testA/test.p'
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
sam2_model = evf_sam2_attacker.from_pretrained(evf_sam2_checkpoint, low_cpu_mem_usage=True, **kwargs)
sam2_model.cuda()
sam2_model.eval()
train_set = pickle.load(open(train_set_path, "rb"))
test_set = pickle.load(open(test_set_path, "rb"))
log_sam1_dir = './log_recoco_testA/evfsam1 asr=0.3/coattack/output.json'
example_dir = './sam1_adv_examples(iter:100,loss:-10)recoco_testA/coattack'
if __name__ == "__main__":
    log1 = []
    for i in tqdm.tqdm(range(len(test_set['images']))):

        image = test_set['images'][i]
        current_log1 = {'image_id': int, 'iou_foreground_list': [], 'iou_background_list': [], 'is_success': []}

        image_path = image['file_name']
        image_id = image['id']
        adv_image_path = get_adv_image_path(image_path, example_dir)
        current_log1['image_id'] = image_id
        ground_truth_list = []

        adv_cv2_img = load_np_image(adv_image_path)
        prompt_list = []
        for refs in test_set['img2refs'][image_id]:
            ann_ids = refs['ann_id']
            annotation = test_set['annotations'][ann_ids]
            segmentation = annotation['segmentation'][0]
            ground_truth = to_bool_mask(segmentation, ori_shape=(image['height'], image['width']), adv_shape=adv_cv2_img.shape[:2])
            for ref in refs['sentences']:
                prompt_list.append(ref['sent'])
                ground_truth_list.append(ground_truth)
        if len(prompt_list) == 0:
            prompt_list.append(train_set['img2refs'][image_id][0]['sentences'][0]['sent'])
            ground_truth_list.append(ground_truth)
        with torch.no_grad():
            adv_masks1 = sam1_model.test(adv_cv2_img, prompt_list, tokenizer)[0]

        output_list1 = (adv_masks1 > 0)

        record_iou_ratio(current_log1, output_list1, ground_truth_list)
        log1.append(current_log1)
        torch.cuda.empty_cache()
    with open(log_sam1_dir, 'w', encoding='utf-8') as json_file:
        json.dump(log1, json_file, indent=4)  # 写入 JSON 数据