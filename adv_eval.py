import argparse
import json
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import tqdm
import cv2
import numpy as np
import torch
print(torch.__version__)
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForMaskedLM
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (AverageMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)
from evf_sam_attacker import evf_sam_attacker
from evf_sam2_attacker import evf_sam2_attacker
import pickle
import random
import re

device = 'cuda:0'
def plot_loss(loss_list):
    plt.plot(range(1, 100 + 1), loss_list)
    plt.xlabel('adv iters')
    plt.ylabel('Loss')
    plt.title('mask Loss Over iters')
    plt.grid(True)
    fig = plt.gcf()  # gcf() 获取当前的 Figure
    plt.close(fig)  # 关闭当前图表，防止重复显示
    return fig
def caculate_aver_loss(total_list):
    averge_loss_list = []
    for i in range(100):
        total = 0
        for j in range(len(total_list)):
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

def load_np_image(image_path):
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    return image_np
mlm_checkpoint = './check points/xlm-roberta-base'
evf_sam2_checkpoint = './check points/evf-sam2'
evf_sam_checkpoint = './check points/evf-sam'
train_set_path = './refcoco+/train.p'
test_set_path = './reccoco/testA/test.p'
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
mlm_model = AutoModelForMaskedLM.from_pretrained(mlm_checkpoint).to(device).eval()
mlm_tok = AutoTokenizer.from_pretrained(mlm_checkpoint, use_fast=False)
# sam2_model =evf_sam2_attacker.from_pretrained(evf_sam2_checkpoint, low_cpu_mem_usage=True, **kwargs)
# sam2_model.cuda()
# sam2_model.eval()
train_set = pickle.load(open(train_set_path, "rb"))
test_set = pickle.load(open(test_set_path, "rb"))


save_dir = '/public/chenxingbai/chenxingbai/EVF-SAM-main/sam1_adv_examples_recoco+/CoAttack'
if __name__ == "__main__":




    total_list = []
    for i in tqdm.tqdm(range(len(train_set['images']))):
    # for image in tqdm.tqdm(train_set['images']):

        image = train_set['images'][i]
        image_path = image['file_name']
        image_id = image['id']

        # prompt = train_set['img2refs'][image_id][0]['sentences'][0]['sent']
        prompt_list = []
        for refs in train_set['img2refs'][image_id]:
            for ref in refs['sentences']:
                prompt_list.append(ref['sent'])
        prompt = prompt_list[0]
        adv_cv2_img, loss_list = sam1_model.coattack(image_path, prompt, tokenizer, mlm_tok, mlm_model)
        total_list.append(loss_list)
        # save_dir = get_adv_save_dir(image_path, sam2_adv_dir, 'ours')
        save_img_dir = get_adv_save_dir1(image_path, save_dir)
        save_img = cv2.cvtColor(adv_cv2_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_img_dir, save_img)
        # fig.savefig(fig_path)
        # plt.close(fig)
        torch.cuda.empty_cache()

    torch.cuda.empty_cache()
