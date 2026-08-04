import glob
import os
import random
import copy
import pickle
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as maskUtils
import random
from model.segment_anything.utils.transforms import ResizeLongestSide
from refer import REFER
from torchvision import transforms
import json
from PIL import Image
from torchvision.transforms.functional import resize, to_pil_image


output_dir = '/public/chenxingbai/chenxingbai/EVF-SAM-main/splited_dataset_new/reccoco/testA_randon42'

base_image_dir = '/public/chenxingbai/chenxingbai/EVF-SAM-main/dataset/refer_seg'
ds = 'refcocog'
splitby = 'umd'
split = 'testA'
random.seed(42)
def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"数据已保存到 {file_path}")

def save_to_pickle(data, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    print(f"数据已保存为 {file_path}")

if __name__ == '__main__':
    refer_api = REFER(base_image_dir, ds, splitby)
    ref_ids_val = refer_api.getRefIds(split=split)
    images_ids_val = refer_api.getImgIds(ref_ids=ref_ids_val)
    refs_val = refer_api.loadRefs(ref_ids=ref_ids_val)
    refer_seg_ds = {}
    refer_seg_ds["images"] = []
    loaded_images = refer_api.loadImgs(image_ids=images_ids_val)
    for item in loaded_images:
        item = item.copy()
        if ds == "refclef":
            item["file_name"] = os.path.join(
                base_image_dir, "images/saiapr_tc-12", item["file_name"]
            )
        elif ds in ["refcoco", "refcoco+", "refcocog", "grefcoco"]:
            item["file_name"] = os.path.join(
                base_image_dir,
                "images/mscoco/images/train2014",
                item["file_name"],
            )
        refer_seg_ds["images"].append(item)
    refer_seg_ds["annotations"] = refer_api.Anns  # anns_val

    img2refs = {}
    for ref in refs_val:
        image_id = ref["image_id"]
        img2refs[image_id] = img2refs.get(image_id, []) + [
            ref,
        ]
    refer_seg_ds["img2refs"] = img2refs
    train_set = copy.deepcopy(refer_seg_ds)
    test_set = copy.deepcopy(refer_seg_ds)
    image2ref = refer_seg_ds["img2refs"]
    for image_id, refs in image2ref.items():
        for i in range(len(refs)):
            sentences = image2ref[image_id][i]['sentences']
            train_sentence_index = random.randint(0, len(sentences) - 1)
            train_set["img2refs"][image_id][i]['sentences'] = [sentences[train_sentence_index]]
            # 将剩余句子作为测试集句子
            test_set["img2refs"][image_id][i]['sentences'] = (
                    sentences[:train_sentence_index] + sentences[train_sentence_index + 1:]
            )
    save_to_pickle(train_set, os.path.join(output_dir, 'train.p'))
    save_to_pickle(test_set, os.path.join(output_dir, 'test.p'))
    data_type = "refer_seg"