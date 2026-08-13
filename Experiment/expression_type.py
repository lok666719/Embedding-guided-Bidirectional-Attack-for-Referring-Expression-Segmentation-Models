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
from collections import defaultdict

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

spatial_keywords = [
    "left", "right", "top", "bottom", "middle", "center", "front", "back",
    "near", "nearest", "far", "farthest", "next to", "in the middle",
    "on the left", "on the right"
]

relational_keywords = [
    "with", "holding", "wearing", "carrying", "behind", "under", "over",
    "beside", "in front of", "riding", "sitting on", "standing by",
    "attached to", "covering", "looking at"
]

def load_np_image(image_path):
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    return image_np

def classify_expression(text: str) -> str:
    t = text.lower().strip()

    for kw in spatial_keywords:
        if kw in t:
            return "spatial"

    for kw in relational_keywords:
        if kw in t:
            return "relational"

    return "attribute"
dataset_configs = [
    {
        "name": "refcoco",
        "test_set_path": "/public/chenxingbai/chenxingbai/EVF-SAM-main/splited_dataset_new/reccoco/testA/test.p",
        "eval_json_path": "/public/chenxingbai/chenxingbai/EVF-SAM-main/log_recoco_testA/evfsam1 asr=0.3/ours v1/output.json",
    },
    {
        "name": "refcoco+",
        "test_set_path": "/public/chenxingbai/chenxingbai/EVF-SAM-main/splited_dataset_new/refcoco+/test.p",
        "eval_json_path": "/public/chenxingbai/chenxingbai/EVF-SAM-main/log_refcoco+/evfsam1 asr=0.3/ours v1/output.json",
    },
    {
        "name": "refcocog",
        "test_set_path": "/public/chenxingbai/chenxingbai/EVF-SAM-main/splited_dataset_new/refcocog/all_>=5/test.p",
        "eval_json_path": "/public/chenxingbai/chenxingbai/EVF-SAM-main/log refcocog all>=5 evfsam1/asr=0.3/ours v1/output.json",
    },
]

evf_sam2_checkpoint = '/public/chenxingbai/chenxingbai/EVF-SAM-main/check points/evf-sam2'
evf_sam_checkpoint = '/public/chenxingbai/chenxingbai/EVF-SAM-main/check points/evf-sam'
train_set_path = '/public/chenxingbai/chenxingbai/EVF-SAM-main/splited_dataset_new/reccoco/testA/train.p'
test_set_path = '/public/chenxingbai/chenxingbai/EVF-SAM-main/splited_dataset_new/reccoco/testA/test.p'
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
# sam2_model = evf_sam2_attacker.from_pretrained(evf_sam2_checkpoint, low_cpu_mem_usage=True, **kwargs)
# sam2_model.cuda()
# sam2_model.eval()
# train_set = pickle.load(open(train_set_path, "rb"))
# test_set = pickle.load(open(test_set_path, "rb"))
log_sam1_dir = '/public/chenxingbai/chenxingbai/EVF-SAM-main/log_recoco_testA/evfsam1 asr=0.3/ours v1/output.json'
example_dir = '/public/chenxingbai/chenxingbai/EVF-SAM-main/sam1_adv_examples(iter:100,loss:-10)recoco_testA/ours v1 0.01'
if __name__ == "__main__":
    all_records = []

    for cfg in dataset_configs:
        dataset_name = cfg["name"]
        test_set = pickle.load(open(cfg["test_set_path"], "rb"))
        eval_results = json.load(open(cfg["eval_json_path"], "r", encoding="utf-8"))

        print(f"Processing dataset: {dataset_name}")

        # 建立 test_set 的 image_id 索引
        image_dict = {img["id"]: img for img in test_set["images"]}

        # 以 json 为主循环
        for result in tqdm.tqdm(eval_results):
            image_id = result["image_id"]

            if image_id not in image_dict:
                print(f"[Warning] image_id={image_id} in {dataset_name} eval json not found in test_set")
                continue

            image = image_dict[image_id]

            iou_fg_list = result["iou_foreground_list"]
            iou_bg_list = result["iou_background_list"]
            success_list = result["is_success"]

            prompt_list = []
            for refs in test_set["img2refs"][image_id]:
                for ref in refs["sentences"]:
                    prompt_list.append(ref["sent"])

            # 安全检查
            if not (len(prompt_list) == len(iou_fg_list) == len(iou_bg_list) == len(success_list)):
                print(f"[Warning] Length mismatch for dataset={dataset_name}, image_id={image_id}")
                print(
                    f"  prompts={len(prompt_list)}, fg={len(iou_fg_list)}, bg={len(iou_bg_list)}, suc={len(success_list)}")
                continue

            for k, prompt in enumerate(prompt_list):
                all_records.append({
                    "dataset": dataset_name,
                    "image_id": image_id,
                    "prompt": prompt,
                    "expr_type": classify_expression(prompt),
                    "iou_foreground": float(iou_fg_list[k]),
                    "iou_background": float(iou_bg_list[k]),
                    "is_success": bool(success_list[k]),
                })


    # ========= 4. 分组统计 =========
    def summarize(records, by_key, ordered_keys=None):
        grouped = defaultdict(list)
        for item in records:
            grouped[item[by_key]].append(item)

        if ordered_keys is None:
            keys = sorted(grouped.keys())
        else:
            keys = ordered_keys

        summary = []
        for key in keys:
            items = grouped.get(key, [])
            if len(items) == 0:
                summary.append({
                    by_key: key,
                    "num_samples": 0,
                    "ASR@30": None,
                    "mIoU-GT": None,
                    "mIoU-UTM": None,
                })
                continue

            n = len(items)
            asr = 100.0 * sum(x["is_success"] for x in items) / n
            miou_gt = sum(x["iou_foreground"] for x in items) / n
            miou_utm = sum(x["iou_background"] for x in items) / n

            summary.append({
                by_key: key,
                "num_samples": n,
                "ASR@30": round(asr, 2),
                "mIoU-GT": round(miou_gt, 2),
                "mIoU-UTM": round(miou_utm, 2),
            })
        return summary


    overall_expr_summary = summarize(
        all_records,
        by_key="expr_type",
        ordered_keys=["attribute", "spatial", "relational"]
    )

    print("\n=== Overall expression-type summary across all datasets ===")
    for row in overall_expr_summary:
        print(row)

    # ========= 5. 可选：分数据集统计 =========
    dataset_expr_summary = {}
    for dataset_name in ["refcoco", "refcoco+", "refcocog"]:
        subset = [x for x in all_records if x["dataset"] == dataset_name]
        dataset_expr_summary[dataset_name] = summarize(
            subset,
            by_key="expr_type",
            ordered_keys=["attribute", "spatial", "relational"]
        )

    print("\n=== Per-dataset expression-type summary ===")
    for dataset_name, rows in dataset_expr_summary.items():
        print(f"\n[{dataset_name}]")
        for row in rows:
            print(row)

    # ========= 6. 保存 =========
    with open("all_expression_records.json", "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    with open("overall_expr_summary.json", "w", encoding="utf-8") as f:
        json.dump(overall_expr_summary, f, ensure_ascii=False, indent=2)

    with open("dataset_expr_summary.json", "w", encoding="utf-8") as f:
        json.dump(dataset_expr_summary, f, ensure_ascii=False, indent=2)