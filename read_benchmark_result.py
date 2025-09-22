import json
import numpy as np
import os
path1 = './log_recoco_testA/evfsam1 asr=0.3/coattack/output.json'

result = {'IoU_background': int, 'IoU_foreground': int, 'ASR': int}

def change_filename_to_result(file_path):
    """
    将文件路径中的文件名更改为'result'，扩展名保持不变。
    :param file_path: 原始文件路径
    :return: 文件名更改后的新路径
    """
    # 分离路径与文件名
    dir_path, file_name = os.path.split(file_path)
    # 获取原始文件的扩展名
    _, file_extension = os.path.splitext(file_name)
    # 构造新的文件路径
    new_file_path = os.path.join(dir_path, f"result{file_extension}")
    return new_file_path
path2 = change_filename_to_result(path1)

with open(path1, 'r') as f:
    data = json.load(f)
    iou_background_list = []
    iou_foreground_list = []
    asr_list = []
    for i in data:
        iou_foreground_list += i['iou_foreground_list']
        iou_background_list += i['iou_background_list']
        asr_list += i['is_success']
    true_count = sum(asr_list)
    asr = true_count / len(asr_list)
    averge_foreground_iou = np.mean(iou_foreground_list)
    averge_background_iou = np.mean(iou_background_list)
    asr = round(asr, 5)
result['IoU_background'] = averge_background_iou
result['IoU_foreground'] = averge_foreground_iou
result['ASR'] = asr
with open(path2, 'w', encoding='utf-8') as json_file:
    json.dump(result, json_file, indent=4)  # 写入 JSON 数据