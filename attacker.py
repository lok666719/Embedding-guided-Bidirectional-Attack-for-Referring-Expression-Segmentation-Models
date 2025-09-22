from typing import List
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, BitsAndBytesConfig
from model.segment_anything.utils.transforms import ResizeLongestSide
from model.evf_sam import EvfSamModel
import cv2

from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import re
import random
from torch_dct import dct_2d, idct_2d
def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss

_FILTER_WORDS = set([
    'a','about','above','across','after','afterwards','again','against','ain','all','almost','alone','along','already',
    'also','although','am','among','amongst','an','and','another','any','anyhow','anyone','anything','anyway','anywhere',
    'are','aren',"aren't",'around','as','at','back','been','before','beforehand','behind','being','below','beside',
    'besides','between','beyond','both','but','by','can','cannot','could','couldn',"couldn't",'d','didn',"didn't",
    'doesn',"doesn't",'don',"don't",'down','due','during','either','else','elsewhere','empty','enough','even','ever',
    'everyone','everything','everywhere','except','first','for','former','formerly','from','hadn',"hadn't",'hasn',
    "hasn't",'haven',"haven't",'he','hence','her','here','hereafter','hereby','herein','hereupon','hers','herself','him',
    'himself','his','how','however','hundred','i','if','in','indeed','into','is','isn',"isn't",'it',"it's",'its','itself',
    'just','latter','latterly','least','ll','may','me','meanwhile','mightn',"mightn't",'mine','more','moreover','most',
    'mostly','must','mustn',"mustn't",'my','myself','namely','needn',"needn't",'neither','never',' nevertheless','next',
    'no','nobody','none','noone','nor','not','nothing','now','nowhere','o','of','off','on','once','one','only','onto',
    'or','other','others','otherwise','our','ours','ourselves','out','over','per','please','s','same','shan',"shan't",
    'she',"she's","should've",'shouldn',"shouldn't",'somehow','something','sometime','somewhere','such','t','than','that',
    "that'll",'the','their','theirs','them','themselves','then','thence','there','thereafter','thereby','therefore',
    'therein','thereupon','these','they','this','those','through','throughout','thru','thus','to','too','toward',
    'towards','under','unless','until','up','upon','used','ve','was','wasn',"wasn't",'we','were','weren',"weren't'",'what',
    'whatever','when','whence','whenever','where','whereafter','whereas','whereby','wherein','whereupon','wherever',
    'whether','which','while','whither','who','whoever','whole','whom','whose','why','with','within','without','won',
    "won't",'would','wouldn',"wouldn't",'y','yet','you',"you'd","you'll","you're","you've",'your','yours','yourself',
    'yourselves','.','-','a the','/','?','some','"',',','b','&','!','@','%','^','*','(',')','+','=','<','>','|',':',';','～','·'
])

class evf_sam_attacker(EvfSamModel):
    def __init__(self, config, extra_param=None, **kwargs):
        super(evf_sam_attacker, self).__init__(config, **kwargs)
        self.adv_epsilon = 16
        self.adv_epsilon_text = 0.2
        self.adv_epsilon_image_evf = 4/255
        self.adv_alpha = 2
        self.adv_alpha_text = 0.01
        self.adv_alpha_image_evf = 2/255
        self.adv_iters = 50
        self.select_feature_layers = [4, 5, 6, 7]
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53])
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375])
        self.adv_epsilon_uad = 8 / 255.0
        self.adv_alpha_uad = 2 /255.0
        self.adv_rho = 0.1
        self.adv_st_iters = 5
        self.adv_sigma = 0.01
        # self.attack_device = device
        # self.device = device
        self.mask_threshold = 0.01
        self.alpha = 10.0
        self.alpha_crit = 1
        self.alpha_ncrit = 2
        self.eps_crit = 16
        self.eps_ncrit = 8
    def load_np_image(self, image_path):
        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        return image_np

    def swap_patches_in_image_different(self, batch_images, num_patches, num_swaps, trigger_prob=0.5):
        # 随机决定是否执行函数
        if torch.rand(1).item() >= trigger_prob:
            return batch_images

        B, C, H, W = batch_images.shape

        # 计算每个patch的大小
        patch_size_h = H // int(num_patches ** 0.5)
        patch_size_w = W // int(num_patches ** 0.5)

        # 将整个批次的图像划分为多个patch
        patches = batch_images.unfold(2, patch_size_h, patch_size_h).unfold(3, patch_size_w, patch_size_w)
        patches = patches.contiguous().view(B, C, -1, patch_size_h,
                                            patch_size_w)  # shape: (B, C, num_patches, patch_size_h, patch_size_w)

        # 创建一个与原始批次大小相同的张量来存储交换后的结果
        swapped_batch = torch.zeros_like(batch_images)

        for b in range(B):
            # 随机选择要交换的patch的索引
            swap_indices = random.sample(range(num_patches), num_swaps)

            for i in range(num_patches):

                if i in swap_indices:
                    index = swap_indices.index(i)
                    # 确保索引不超出列表范围
                    if index < len(swap_indices) - 1:
                        new_idx = swap_indices[index + 1]
                    else:
                        new_idx = swap_indices[0]
                else:
                    new_idx = i

                # 计算在原始图像中的位置
                row_idx = i // (H // patch_size_h)
                col_idx = i % (W // patch_size_w)
                row_start = row_idx * patch_size_h
                col_start = col_idx * patch_size_w

                # 从原始图像复制patch到交换后的位置
                swapped_batch[b, :, row_start:row_start + patch_size_h, col_start:col_start + patch_size_w] = patches[b, :, new_idx, :, :]

        return swapped_batch

    def flip(self, x):
        flip_rate = 0.5


        # 翻转图像
        if torch.rand(1) < flip_rate:
            x = torch.flip(x, dims=[-1])  # 水平翻转

        return x

    def dct2d(self, x):
        return dct_2d(x, norm='ortho')  # 使用torch_dct的二维DCT

    def idct2d(self, x):
        return idct_2d(x, norm='ortho')  # 使用torch_dct的二维逆DCT

    def input_diversity(self, x):
        resize_rate = 0.9
        flip_rate = 0.9
        diversity_prob = 0.5
        img_size = x.shape[-1]
        img_resize = int(img_size * resize_rate)

        if resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        # 翻转图像
        if torch.rand(1) < flip_rate:
            x = torch.flip(x, dims=[-1])  # 水平翻转

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1).item() < diversity_prob else x

    def beit3_preprocess(self, x: np.ndarray, img_size=224) -> torch.Tensor:
        '''
        preprocess for BEIT-3 model.
        input: ndarray
        output: torch.Tensor
        '''
        beit_preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC, antialias=None),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        return beit_preprocess(x)

    def beit3_preprocess_1(self, x, img_size=224) -> torch.Tensor:
        '''
        preprocess for BEIT-3 model.
        input: ndarray
        output: torch.Tensor
        '''
        tensor_normed = (x / 255.0)
        return tensor_normed

    def beit3_preprocess_2(self, x: np.ndarray, img_size=224) -> torch.Tensor:
        '''
        preprocess for BEIT-3 model.
        input: ndarray
        output: torch.Tensor
        '''
        beit_preprocess = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC, antialias=None),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        return beit_preprocess(x).to(dtype=torch.float16, device='cuda')

    def sam_preprocess_1(self,
            x: np.ndarray,
            pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
            pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
            img_size=1024,
            model_type="ori") -> torch.Tensor:
        '''
        preprocess of Segment Anything Model, including scaling, normalization and padding.
        preprocess differs between SAM and Effi-SAM, where Effi-SAM use no padding.
        input: ndarray
        output: torch.Tensor
        '''
        assert img_size == 1024, \
            "both SAM and Effi-SAM receive images of size 1024^2, don't change this setting unless you're sure that your employed model works well with another size."

        # Normalize colors
        if model_type == "ori" or model_type == "attack" or model_type == "attack1":
            x = ResizeLongestSide(img_size).apply_image(x)
            h, w = resize_shape = x.shape[:2]
            x = torch.from_numpy(x).permute(2, 0, 1).contiguous()
            # x = (x - pixel_mean) / pixel_std
            # # Pad
            # padh = img_size - h
            # padw = img_size - w
            # x = F.pad(x, (0, padw, 0, padh))
        return x, resize_shape

    def sam_preprocess_2(
            self,
            x,
            resize_shape,
            pixel_mean=torch.tensor([123.675, 116.28, 103.53], device='cuda').view(-1, 1, 1),
            pixel_std=torch.tensor([58.395, 57.12, 57.375], device='cuda').view(-1, 1, 1),
            img_size=1024,
            model_type="ori") -> torch.Tensor:
        '''
        preprocess of Segment Anything Model, including scaling, normalization and padding.
        preprocess differs between SAM and Effi-SAM, where Effi-SAM use no padding.
        input: ndarray
        output: torch.Tensor
        '''
        assert img_size == 1024, \
            "both SAM and Effi-SAM receive images of size 1024^2, don't change this setting unless you're sure that your employed model works well with another size."

        # Normalize colors
        if model_type == "ori" or model_type == "attack" or model_type == "attack1":
            # x = ResizeLongestSide(img_size).apply_image(x)
            # h, w = resize_shape = x.shape[:2]
            # x = torch.from_numpy(x).permute(2, 0, 1).contiguous()
            x = (x - pixel_mean) / pixel_std
            # Pad
            h, w = resize_shape
            padh = img_size - h
            padw = img_size - w
            x = F.pad(x, (0, padw, 0, padh))
        return x
    def de_preprocess(self, x, h, w) -> torch.Tensor:
        x = x[:, :, :h, :w]
        x = x * self.visual_model.pixel_std + self.visual_model.pixel_mean
        return x

    def get_cv2_from_torch(self, torch_img, original_size):
        # h,w = input_size
        # torch_img = self.de_preprocess(img_torchtensor, h, w)
        numpy_img = torch_img.cpu()[0].numpy().astype(np.uint8)
        cv2_img = np.transpose(numpy_img, (1, 2, 0))
        # cv2_img = cv2.resize(cv2_img, (original_size[1], original_size[0]))
        return cv2_img

    def get_cv2_from_torch2(self, img_torchtensor, original_size, resize_shape):
        h, w = resize_shape
        torch_img = self.de_preprocess(img_torchtensor, h, w)
        numpy_img = torch_img.cpu()[0].numpy().astype(np.uint8)
        cv2_img = np.transpose(numpy_img, (1, 2, 0))
        cv2_img = cv2.resize(cv2_img, (original_size[1], original_size[0]))
        return cv2_img

    def sample_points(self, original_size, mask=None, sample_size=10):
        # mask is a two dimensional numpy array (binary mask over original image size)
        if mask is None: # sample in the whole image
            mask = np.ones(original_size, dtype=np.uint8)
        inmask_pixel_positions = np.flip(np.argwhere(mask == True), axis=1)
        sample_size = min(sample_size, inmask_pixel_positions.shape[0])
        sampled_pixel_id = np.random.choice(inmask_pixel_positions.shape[0], sample_size, replace=False)
        sampled_pixel_pos = inmask_pixel_positions[sampled_pixel_id]
        return sampled_pixel_pos

    def transform_coord(self, original_size, points, boxes):
        # points: numpy array of shape (N, 2)
        # boxes: numpy array of shape (M, 4)
        coords_torch, labels_torch, box_torch = None, None, None
        if points is not None:
            labels = np.array([1] * len(points))
            points_coords = ResizeLongestSide(1024).apply_coords(points, original_size)
            coords_torch = torch.as_tensor(points_coords, dtype=torch.float, device='cuda')
            labels_torch = torch.as_tensor(labels, dtype=torch.int, device='cuda')
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]

        points_torch = None if coords_torch is None else (coords_torch, labels_torch)
        return points_torch

    def spectrum_transform(self, x, rho=0.1, sigma=0.01):
        noise = torch.randn_like(x) * sigma
        x_noisy = x + noise
        x_dct = self.dct2d(x_noisy)
        M = torch.empty_like(x_dct).uniform_(1 - rho, 1 + rho)
        x_dct_perturbed = x_dct * M
        return self.idct2d(x_dct_perturbed)

    def SRA_sample_points(self, original_size, mask=None, interval=50):
        if mask is None:
            mask = np.ones(original_size, dtype=np.uint8)

        # 获取掩码的尺寸
        height, width = mask.shape

        # 生成均匀网格坐标
        x_coords = np.arange(0, width, interval)  # 宽度方向上的坐标
        y_coords = np.arange(0, height, interval)  # 高度方向上的坐标

        # 创建所有网格点的坐标组合
        grid_coords = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)

        # 过滤出在掩码中为 True 的点
        valid_points = [point for point in grid_coords if mask[point[1], point[0]]]

        return np.array(valid_points)

    def forward(
            self,
            images: torch.FloatTensor,
            images_evf: torch.FloatTensor,
            input_ids: torch.LongTensor,
            attention_masks: torch.LongTensor,
            offset: torch.LongTensor,
            masks_list: List[torch.FloatTensor],
            label_list: List[torch.Tensor],
            resize_list: List[tuple],
            inference: bool = False,
            **kwargs,
    ):
        original_clean_image = images.data
        batch_size = 1
        assert batch_size == len(offset) - 1

        images_evf_list = []
        for i in range(len(offset) - 1):
            start_i, end_i = offset[i], offset[i + 1]
            images_evf_i = (
                images_evf[i]
                .unsqueeze(0)
                .expand(end_i - start_i, -1, -1, -1)
                .contiguous()
            )
            images_evf_list.append(images_evf_i)
        images_evf = torch.cat(images_evf_list, dim=0)

        multimask_output = False
        output = self.mm_extractor.beit3(
            visual_tokens=images_evf,
            textual_tokens=input_ids,
            text_padding_position=~attention_masks
        )

        feat = output["encoder_out"][:, :1, ...]

        feat = self.text_hidden_fcs[0](feat)
        feat = torch.split(feat, [offset[i + 1] - offset[i] for i in range(len(offset) - 1)])

        pred_masks = []
        for i in range(len(feat)):
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=feat[i],
            )
            sparse_embeddings = sparse_embeddings.to(feat[i].dtype)
            for _ in range(self.adv_iters):
                images.requires_grad = True
                image_embeddings = self.visual_model.image_encoder(self.sam_preprocess_norm(images, resize_list[i]))
                low_res_masks, _ = self.visual_model.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                print('>0 :', (low_res_masks > 0).sum())
                target_low_res_masks = torch.clamp(low_res_masks.detach(), max=-20)
                loss = torch.nn.MSELoss()(low_res_masks, target_low_res_masks)
                grad = torch.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]
                print(f'mask loss:{loss}')
                perturbation = self.adv_alpha * grad.data.sign()
                adv_image_unclipped = images.data - perturbation
                clipped_perturbation = torch.clamp(adv_image_unclipped - original_clean_image,
                                                   min=-self.adv_epsilon,
                                                   max=self.adv_epsilon)
                images = torch.clamp(original_clean_image + clipped_perturbation,
                                     min=0,
                                     max=255).detach()

        pred_mask = self.visual_model.postprocess_masks(
            low_res_masks,
            input_size=resize_list[i],
            original_size=label_list[i].shape,
        )
        pred_masks.append(pred_mask[:, 0])

        gt_masks = masks_list

        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
            }

        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]

            assert (
                    gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            mask_bce_loss += (
                    sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0]
            )
            mask_dice_loss += (
                    dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        loss = mask_loss

        return {
            "loss": loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }

    def segpgd_single_p(self, image_path, prompt, tokenizer):
        image_np = self.load_np_image(image_path)
        original_size = image_np.shape[:2]
        image_beit = self.beit3_preprocess(image_np, 224).to(dtype=torch.float16, device='cuda')
        image_sam_unnormed, resize_shape = self.sam_preprocess_1(image_np)
        image_sam_unnormed = image_sam_unnormed.to(dtype=torch.float16, device='cuda')
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device='cuda')
        images = image_sam_unnormed.unsqueeze(0)
        original_clean_image = images.data
        output = self.mm_extractor.beit3(visual_tokens=image_beit.unsqueeze(0), textual_tokens=input_ids,
                                         text_padding_position=torch.zeros_like(input_ids))
        feat = output["encoder_out"][:, :1, ...]
        feat = self.text_hidden_fcs[0](feat)
        (
            sparse_embeddings,
            dense_embeddings,
        ) = self.visual_model.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
            text_embeds=feat,
        )
        criterion = torch.nn.MSELoss()
        sparse_embeddings = sparse_embeddings.to(feat.dtype)
        loss_list = []
        for i in range(self.adv_iters):
            images.requires_grad = True
            image_embeddings = self.visual_model.image_encoder(self.sam_preprocess_2(images, resize_shape))
            image_embeddings = self.visual_model.image_encoder(self.sam_preprocess_2(images, resize_shape))
            low_res_masks, _ = self.visual_model.mask_decoder(image_embeddings=image_embeddings,
                                                              image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                                                              sparse_prompt_embeddings=sparse_embeddings,
                                                              dense_prompt_embeddings=dense_embeddings,
                                                              multimask_output=False)
            lamb = (i - 1) / (self.adv_iters * 2)
            mask_true = (low_res_masks > 0).int()
            mask_false = (low_res_masks < 0).int()
            print('>0 :', (low_res_masks > 0).sum())
            target_low_res_masks = torch.clamp(low_res_masks.detach(), max=-10)
            loss_t = (1 - lamb) * criterion(mask_true * low_res_masks, target_low_res_masks)
            loss_f = lamb * criterion(mask_false * low_res_masks, target_low_res_masks)
            loss = loss_t + loss_f
            loss_list.append(loss.detach().item())
            grad = torch.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]
            print(f'mask loss:{loss}')
            perturbation = self.adv_alpha * grad.data.sign()
            adv_image_unclipped = images.data - perturbation
            clipped_perturbation = torch.clamp(adv_image_unclipped - original_clean_image,
                                               min=-self.adv_epsilon,
                                               max=self.adv_epsilon)
            images = torch.clamp(original_clean_image + clipped_perturbation,
                                 min=0,
                                 max=255).detach()
        return self.get_cv2_from_torch(images, original_size), loss_list

    def _split_words_and_map(self, text: str):
        """按空格切分并返回 words 与每词在下游 tokenizer 子词序列上的 span（基于 evf tokenizer）。
           keys 中索引基于“去掉 special tokens 的子词序列”。"""
        words = text.split(' ')
        keys = []
        idx = 0
        for w in words:
            subs = self.tokenizer.tokenize(w)
            keys.append([idx, idx + len(subs)])
            idx += len(subs)
        return words, keys

    def _get_mlm_logits(self, input_ids, attention_mask):
        """兼容 older transformers（tuple）与 newer（ModelOutput）"""
        out = self.mlm(input_ids=input_ids, attention_mask=attention_mask)
        return out[0] if isinstance(out, (tuple, list)) else out.logits  # [B, L, V]

    @torch.no_grad()
    def _mlm_candidates_from_masked_string(self, masked_string: str, topk_per_pos=8, max_combos=200, max_return=10,
                                           max_length=64):
        """
        给定一个字符串（其中目标词已替换为 mlm_tokenizer.mask_token），
        在 mlm_tokenizer 下定位 mask token 位置（可能有多个），对这些位置上的 logits 做 top-k，
        然后做 BPE 组合（限制每位 topk_per_pos）并返回最多 max_return 的候选字符串列表。
        """
        device = self.device
        enc = self.mlm_tokenizer(masked_string, padding='max_length', truncation=True, max_length=max_length,
                                 return_tensors='pt')
        input_ids = enc['input_ids'].to(device)
        attn_mask = enc['attention_mask'].to(device)

        # 找到 mask token 在这个 tokenization 下的位置（pos 可多个）
        mask_id = self.mlm_tokenizer.convert_tokens_to_ids(self.mlm_tokenizer.mask_token)
        mask_positions = (input_ids[0] == mask_id).nonzero(as_tuple=True)[0].tolist()
        if len(mask_positions) == 0:
            # 若没有 mask token（极少见），尝试把第一个位置当作目标
            mask_positions = [1]

        logits = self._get_mlm_logits(input_ids, attn_mask)  # [1, L, V]

        # 取每个 mask pos 的 topk ids
        topk_ids_list = []
        for pos in mask_positions:
            vals, ids = logits[0, pos].topk(min(topk_per_pos, logits.size(-1)), dim=-1)
            topk_ids_list.append(ids.tolist())  # list of ids

        # 笛卡尔组合（限制总数）
        combos = [[]]
        for ids in topk_ids_list:
            new = []
            for acc in combos:
                for t in ids:
                    new.append(acc + [int(t)])
            combos = new
            if len(combos) > max_combos:
                combos = combos[:max_combos]

        # 转成字符串并过滤
        final = []
        seen = set()
        for seq in combos:
            toks = [self.mlm_tokenizer.convert_ids_to_tokens(i) for i in seq]
            cand_str = self.mlm_tokenizer.convert_tokens_to_string(toks).strip()
            if not cand_str:
                continue
            # 过滤掉subword前缀（对不同tokenizer前缀差异做宽容处理）
            # 如果全部是 subword 片段（以 '##' 或以 '▁' 无前缀等），我们仍允许 convert_tokens_to_string 进行合并
            if cand_str.lower() in _FILTER_WORDS:
                continue
            if cand_str.lower() in seen:
                continue
            seen.add(cand_str.lower())
            final.append(cand_str)
            if len(final) >= max_return:
                break
        return final

    @torch.no_grad()
    def select_adversarial_text_with_mlm_tokenizers(
            self,
            image_beit,  # [C, H_patch, W_patch] 或者 1xC... (与 coattack 中 image_beit 对齐)
            images,  # [1,3,H,W] 原始 SAM 输入（unnormed tensor），用于评估
            resize_shape,  # sam_preprocess_1 返回的 resize_shape（供 sam_preprocess_2 使用）
            prompt: str,
            mlm_topk_per_pos: int = 6,
            num_perturbation: int = 1,
            batch_size: int = 16,
            max_length: int = 64,
    ):
        """
        使用 self.mlm + self.mlm_tokenizer 生成候选、使用 self.tokenizer (EVF) 评估并挑选替换词。
        返回 adv_text（字符串）与 feat_final（[1,1,Cp] 可直接用于 prompt_encoder）
        """
        device = self.device

        # 内部：给一批文本算排序用的loss（与 coattack 保持一致）
        def batch_rank_loss(text_list):
            # tokenize 用下游 tokenizer（self.tokenizer）
            enc = self.tokenizer(text_list, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
            input_ids = enc['input_ids'].to(device)
            pad_pos = torch.zeros_like(input_ids).bool().to(device)  # 你 coattack 用的是 zeros

            # BEiT3: 用 image_beit（注意 expand/重复）
            # image_beit 传入时应是单张 224 特征（C,H,W）; 需要 unsqueeze then repeat
            visual_tokens = image_beit.unsqueeze(0).repeat(len(text_list), 1, 1, 1)
            out = self.mm_extractor.beit3(visual_tokens=visual_tokens, textual_tokens=input_ids,
                                          text_padding_position=pad_pos)
            feat = out["encoder_out"][:, :1, ...]  # [B,1,C]
            feat = self.text_hidden_fcs[0](feat)  # [B,1,Cp]

            # prompt encoder
            sparse_emb, dense_emb = self.visual_model.prompt_encoder(points=None, boxes=None, masks=None,
                                                                     text_embeds=feat)
            # image path
            # img_batch = images.repeat(len(text_list), 1, 1, 1).to(dtype=images.dtype, device=device)
            img_enc_in = self.sam_preprocess_2(images, resize_shape)
            image_embeddings = self.visual_model.image_encoder(img_enc_in)
            low_res_masks, _ = self.visual_model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )
            # 用 coattack 的排序 loss（与你一致）
            target = torch.clamp(low_res_masks.detach(), max=-10)
            loss = F.mse_loss(low_res_masks, target, reduction='none')  # [B,1,h,w]
            loss = loss.mean(dim=(1, 2, 3))  # [B]
            return loss.detach().tolist()

        # ==== 1) leave-one-out: 用 evf tokenizer 按词替换为下游 mask，评估重要性 ==== #
        words, keys = self._split_words_and_map(prompt)
        mask_token_evf = getattr(self.tokenizer, "mask_token", None) or "[MASK]"
        masked_texts = []
        for i in range(len(words)):
            tmp = copy.deepcopy(words)
            tmp[i] = mask_token_evf
            masked_texts.append(' '.join(tmp))

        importance = []
        for i in range(0, len(masked_texts), batch_size):
            importance += batch_rank_loss(masked_texts[i:i + batch_size])

        # 按重要性降序
        order = sorted(range(len(words)), key=lambda i: importance[i], reverse=True)

        # ==== 2) 对重要词逐个尝试替换（使用 mlm_tokenizer + mlm 生成候选），选能让 loss 最大的替换 ==== #
        final_words = copy.deepcopy(words)
        changed = 0
        for idx in order:
            if changed >= num_perturbation:
                break
            tgt = final_words[idx]
            if tgt.lower() in _FILTER_WORDS:
                continue

            # 生成 masked string 用于 mlm 预测：把该词替为 mlm_tokenizer.mask_token（字符串）
            tmp_words = copy.deepcopy(final_words)
            tmp_words[idx] = self.mlm_tokenizer.mask_token
            masked_string = ' '.join(tmp_words)

            # mlm 生成候选（会返回字符串）
            cands = self._mlm_candidates_from_masked_string(masked_string, topk_per_pos=mlm_topk_per_pos,
                                                       max_length=max_length)
            if len(cands) == 0:
                continue
            # 把原词放在候选首位保证兜底
            if tgt not in cands:
                cands = [tgt] + cands

            # 评估所有候选
            replace_texts = []
            for c in cands:
                tmp = copy.deepcopy(final_words)
                tmp[idx] = c
                replace_texts.append(' '.join(tmp))

            best_loss, best_text = None, None
            for i in range(0, len(replace_texts), batch_size):
                losses = batch_rank_loss(replace_texts[i:i + batch_size])
                for t, lv in zip(replace_texts[i:i + batch_size], losses):
                    if (best_loss is None) or (lv > best_loss):
                        best_loss, best_text = lv, t

            if best_text is not None and best_text != ' '.join(final_words):
                final_words = best_text.split(' ')
                changed += 1

        adv_text = ' '.join(final_words)

        # ==== 3) 计算最终 prompt embedding（与 coattack pipeline 完全相同的格式） ==== #
        enc_final = self.tokenizer(adv_text, padding='max_length', truncation=True, max_length=32,
                                   return_tensors='pt').to(device)
        out_final = self.mm_extractor.beit3(visual_tokens=image_beit.unsqueeze(0),
                                            textual_tokens=enc_final['input_ids'],
                                            text_padding_position=torch.zeros_like(enc_final['input_ids'],
                                                                                   dtype=torch.bool).to(device))
        feat_final = self.text_hidden_fcs[0](out_final["encoder_out"][:, :1, ...])  # [1,1,Cp]

        return adv_text, feat_final
    def coattack(self, image_path, prompt, tokenizer, mlm_tok, mlm):
        self.mlm = mlm
        self.mlm_tokenizer = mlm_tok
        self.tokenizer = tokenizer
        image_np = self.load_np_image(image_path)
        original_size = image_np.shape[:2]
        image_beit = self.beit3_preprocess(image_np, 224).to(dtype=torch.float16, device=self.device)
        image_sam_unnormed, resize_shape = self.sam_preprocess_1(image_np)
        image_sam_unnormed = image_sam_unnormed.to(dtype=torch.float16, device='cuda')
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device='cuda')
        images = image_sam_unnormed.unsqueeze(0)
        original_clean_image = images.data
        # ←—— 新增：先做一次文本“换词”（只在循环外做一次）
        with torch.no_grad():
            adv_text, adv_feat = self.select_adversarial_text_with_mlm_tokenizers(
            image_beit=image_beit, images=images, resize_shape=resize_shape,
            prompt=prompt, max_length=32, batch_size=16, num_perturbation=1
        )
            print(f"[BertAttack-XLM] '{prompt}' → '{adv_text}'")

            clean_output = self.mm_extractor.beit3(visual_tokens=image_beit.unsqueeze(0), textual_tokens=input_ids,
                                             text_padding_position=torch.zeros_like(input_ids))
            feat = clean_output["encoder_out"][:, :1, ...]
            clean_feat = self.text_hidden_fcs[0](feat)
            (
                clean_sparse_embeddings,
                clean_dense_embeddings,
            ) = self.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=clean_feat.detach(),
            )
            (
                adv_sparse_embeddings,
                adv_dense_embeddings,
            ) = self.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=adv_feat.detach(),
            )
            clean_sparse_embeddings = clean_sparse_embeddings.to(clean_feat.dtype).detach()
            clean_image_embeddings = self.visual_model.image_encoder(self.sam_preprocess_2(original_clean_image.detach(), resize_shape))
            adv_t_lowmask, _ = self.visual_model.mask_decoder(image_embeddings=clean_image_embeddings,
                                                              image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                                                              sparse_prompt_embeddings=adv_sparse_embeddings.detach(),
                                                              dense_prompt_embeddings=adv_dense_embeddings.detach(),
                                                              multimask_output=False)
            clean_lowmask, _ = self.visual_model.mask_decoder(image_embeddings=clean_image_embeddings,
                                                           image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                                                           sparse_prompt_embeddings=clean_sparse_embeddings.detach(),
                                                           dense_prompt_embeddings=clean_dense_embeddings.detach(),
                                                           multimask_output=False)
        loss_list = []
        for i in range(self.adv_iters):
            images.requires_grad = True
            image_embeddings = self.visual_model.image_encoder(self.sam_preprocess_2(images, resize_shape))
            low_res_masks, _ = self.visual_model.mask_decoder(image_embeddings=image_embeddings,
                                                              image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                                                              sparse_prompt_embeddings=adv_sparse_embeddings,
                                                              dense_prompt_embeddings=adv_dense_embeddings,
                                                              multimask_output=False)
            print('>0 :', (low_res_masks > 0).sum())
            target_low_res_masks = torch.clamp(low_res_masks.detach(), max=-10)
            loss1 = torch.nn.MSELoss()(low_res_masks, adv_t_lowmask)
            loss2 = torch.nn.MSELoss()(low_res_masks, clean_lowmask)
            loss = loss1 + 2 * loss2
            loss_list.append(loss.detach().item())
            grad = torch.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]
            print(f'mask loss:{loss}')
            perturbation = self.adv_alpha * grad.data.sign()
            adv_image_unclipped = images.data + perturbation
            clipped_perturbation = torch.clamp(adv_image_unclipped - original_clean_image,
                                               min=-self.adv_epsilon,
                                               max=self.adv_epsilon)
            images = torch.clamp(original_clean_image + clipped_perturbation,
                                 min=0,
                                 max=255).detach()
        return self.get_cv2_from_torch(images, original_size), loss_list

    def single_prompt_attack(self, image_path, prompt, tokenizer):
        image_np = self.load_np_image(image_path)
        original_size = image_np.shape[:2]
        image_beit = self.beit3_preprocess(image_np, 224).to(dtype=torch.float16, device='cuda')
        image_sam_unnormed, resize_shape = self.sam_preprocess_1(image_np)
        image_sam_unnormed = image_sam_unnormed.to(dtype=torch.float16, device='cuda')
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device='cuda')
        images = image_sam_unnormed.unsqueeze(0)
        original_clean_image = images.data
        output = self.mm_extractor.beit3(visual_tokens=image_beit.unsqueeze(0), textual_tokens=input_ids,
                                         text_padding_position=torch.zeros_like(input_ids))
        feat = output["encoder_out"][:, :1, ...]
        feat = self.text_hidden_fcs[0](feat)
        (
            sparse_embeddings,
            dense_embeddings,
        ) = self.visual_model.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
            text_embeds=feat,
        )
        sparse_embeddings = sparse_embeddings.to(feat.dtype)
        loss_list = []
        for i in range(self.adv_iters):
            images.requires_grad = True
            image_embeddings = self.visual_model.image_encoder(self.sam_preprocess_2(images, resize_shape))
            low_res_masks, _ = self.visual_model.mask_decoder(image_embeddings=image_embeddings,
                                                              image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                                                              sparse_prompt_embeddings=sparse_embeddings,
                                                              dense_prompt_embeddings=dense_embeddings,
                                                              multimask_output=False)
            print('>0 :', (low_res_masks > 0).sum())
            target_low_res_masks = torch.clamp(low_res_masks.detach(), max=-10)
            loss = torch.nn.MSELoss()(low_res_masks, target_low_res_masks)
            loss_list.append(loss.detach().item())
            grad = torch.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]
            print(f'mask loss:{loss}')
            perturbation = self.adv_alpha * grad.data.sign()
            adv_image_unclipped = images.data - perturbation
            clipped_perturbation = torch.clamp(adv_image_unclipped - original_clean_image,
                                               min=-self.adv_epsilon,
                                               max=self.adv_epsilon)
            images = torch.clamp(original_clean_image + clipped_perturbation,
                                 min=0,
                                 max=255).detach()
        return self.get_cv2_from_torch(images, original_size), loss_list

    def extract_text_cross_attention(self, l_aux, input_ids, multiway_split_position):
        """
        Extracts aggregated attention from text tokens to visual tokens from l_aux.
        :param l_aux: List of dicts, each with 'attn_weights' key.
        :param input_ids: Tensor of token ids, shape [1, L_text]
        :param multiway_split_position: Index separating visual and text tokens
        :return: attn_scores: [L_text], tokens: List[str]
        """
        import torch
        import torch.nn.functional as F

        attn_list = []

        for layer in l_aux:
            if layer is None or "attn_weights" not in layer:
                continue
            attn = layer["attn_weights"]  # [B, H, T, T]
            # 取出文本→图像的 cross attention（行是text，列是image）
            cross_attn = attn[:, :, multiway_split_position:, :multiway_split_position].permute(1, 0, 2, 3)  # [B, H, L_text, L_image]
            # 聚合 heads 和 image 维度，得到对每个文本 token 的聚合注意力值
            cross_attn_per_text = cross_attn.mean(dim=1).sum(dim=-1)  # [B, L_text]
            attn_list.append(cross_attn_per_text)

        # 所有层的 attention 求平均
        all_layers_attn = torch.stack(attn_list, dim=0)  # [L, B, L_text]
        mean_attn = all_layers_attn.mean(dim=0)[0]  # [L_text]

        return mean_attn, input_ids[0]  # 返回 token id 列表

    def fused_repr_from_hs(self, hs, num_mask_tokens, mode="iou+mask"):
        # hs: [B, 1 + num_mask_tokens + N_prompt, C]
        if mode == "iou+mask":
            # 只取 iou + mask tokens
            x = hs[:, :1 + num_mask_tokens, :]  # [B, 1+M, C]
        else:
            # 或者取所有 tokens（包括 prompt tokens）
            x = hs
        # 池化得到一个向量
        e = x.mean(dim=1)  # [B, C]
        e = torch.nn.functional.normalize(e, dim=-1)
        return e

    def _detok_sentencepiece(self, evf_tokens):
        s = "".join(t.replace("▁", " ") if t not in ("<s>", "</s>") else "" for t in evf_tokens).strip()
        return " ".join(s.split())

    def _extract_word_string_from_evf_tokens(self, evf_tokens, word_pos):
        assert evf_tokens[word_pos].startswith("▁")
        pieces = [evf_tokens[word_pos][1:]]
        i = word_pos + 1
        while i < len(evf_tokens) and (not evf_tokens[i].startswith("▁")) and evf_tokens[i] not in ("<s>", "</s>"):
            pieces.append(evf_tokens[i])
            i += 1
        return "".join(pieces)

    def _mask_one_word(self, evf_tokens, word_pos):
        out = evf_tokens[:]
        out[word_pos] = "<mask>"
        i = word_pos + 1
        while i < len(out) and (not out[i].startswith("▁")) and out[i] not in ("<s>", "</s>"):
            out[i] = ""
            i += 1
        out = [t for t in out if t != ""]
        return self._detok_sentencepiece(out)

    def _replace_word_in_sentence(self, orig_sentence, orig_word, new_word):
        pat = re.compile(rf"\b{re.escape(orig_word)}\b", flags=re.IGNORECASE)
        new_sent, n = pat.subn(new_word, orig_sentence, count=1)
        if n == 0:
            parts = orig_sentence.split()
            for i, p in enumerate(parts):
                if p.lower() == orig_word.lower():
                    parts[i] = new_word
                    break
            new_sent = " ".join(parts)
        return new_sent

    # ========== MLM 候选：用 self.mlm_tok / self.mlm_model ==========
    def _is_clean_mlm_token(self, tok: str):
        # 只保留整词英文（去停用词）
        EN_STOP = {"the", "a", "an", "of", "to", "and", "in", "on", "for", "with", "at", "by", "from",
                   "as", "is", "are", "was", "were", "be", "been", "being", "that", "this", "it",
                   "its", "or", "but", "if", "than", "then", "so", "not", "no", "do", "did", "does",
                   "doing", "can", "could", "would", "should", "may", "might", "will", "shall",
                   "have", "has", "had"}
        if not tok.startswith("▁"): return False
        w = tok[1:]
        if len(w) < 2: return False
        if not re.fullmatch(r"[A-Za-z\-]+", w): return False
        if w.lower() in EN_STOP: return False
        return True

    @torch.no_grad()
    def _mlm_candidates_by_string(self, masked_sentence: str, topk: int = 100):
        mlm_tok = self.mlm_tok
        mlm_model = self.mlm_model
        device = next(mlm_model.parameters()).device
        inp = mlm_tok(masked_sentence, return_tensors="pt").to(device)
        mask_idx = (inp.input_ids[0] == mlm_tok.mask_token_id).nonzero(as_tuple=True)[0]
        logits = mlm_model(**inp).logits[0, mask_idx]  # [1, vocab]
        ids = torch.topk(logits, k=topk, dim=-1).indices[0].tolist()

        cands, seen = [], set()
        for tid in ids:
            tok = mlm_tok.convert_ids_to_tokens([tid])[0]
            if self._is_clean_mlm_token(tok):
                w = tok[1:]
                wl = w.lower()
                if wl not in seen:
                    seen.add(wl)
                    cands.append(w)
                if len(cands) >= 10:
                    break
        return cands  # ['child','man','dog',...]

    # ========== 评估候选：回到你的模型，算融合向量余弦，选最优 ==========
    @torch.no_grad()
    def _get_fused_from_text_ids(self, image_embeddings, input_ids, image_beit):
        # 走你已有的 beit3→prompt encoder→mask decoder，返回融合表示 e
        output = self.mm_extractor.beit3(visual_tokens=image_beit.unsqueeze(0), textual_tokens=input_ids,
                                         text_padding_position=torch.zeros_like(input_ids))
        feat = output["encoder_out"][:, :1, ...]
        feat = self.text_hidden_fcs[0](feat)
        (
            sparse_embeddings,
            dense_embeddings,
        ) = self.visual_model.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
            text_embeds=feat,
        )
        # 一次 forward mask decoder 拿 hs
        _, _, hs = self.visual_model.mask_decoder(image_embeddings=image_embeddings,
                                                              image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                                                              sparse_prompt_embeddings=sparse_embeddings,
                                                              dense_prompt_embeddings=dense_embeddings,
                                                              multimask_output=False,
                                                              return_hs=True)
        e = self.fused_repr_from_hs(hs.detach(), self.visual_model.mask_decoder.num_mask_tokens).detach()
        return e, sparse_embeddings, dense_embeddings  # 返回 e 以及可用于后续 PGD 的 prompt embeds

    @torch.no_grad()
    def _pick_best_text_replacement(self, input_ids_1xL, tokenizer, image_embeddings_clean, image_beit, e_o):
        # 1) 文本注意力：用你已有的 extract_text_cross_attention 选词首位置
        beit_out = self.mm_extractor.beit3(visual_tokens=image_beit.unsqueeze(0), textual_tokens=input_ids_1xL,
                                         text_padding_position=torch.zeros_like(input_ids_1xL))
        msp = beit_out["multiway_split_position"]
        l_aux = beit_out["l_aux"]
        attn_scores, token_ids = self.extract_text_cross_attention(l_aux, input_ids_1xL, msp)
        evf_tokens = tokenizer.convert_ids_to_tokens(token_ids)

        # 只在词首（▁）里选最大注意力的一个位置（ϵt=1）
        word_starts = [i for i, t in enumerate(evf_tokens) if t.startswith("▁")]
        if not word_starts:
            return None
        # 将 attn_scores（对齐 token_ids）限制在词首位置再取 argmax
        scores_on_starts = [(i, float(attn_scores[i].item())) for i in word_starts]
        word_pos = max(scores_on_starts, key=lambda x: x[1])[0]

        # 2) 构造 <mask> 句子 → MLM 候选
        orig_sentence = self._detok_sentencepiece(evf_tokens)
        orig_word = self._extract_word_string_from_evf_tokens(evf_tokens, word_pos)
        masked_sentence = self._mask_one_word(evf_tokens, word_pos)
        cand_words = self._mlm_candidates_by_string(masked_sentence)
        if not cand_words:
            return None
            # 3) 原文的融合表示 e_o

        # 4) 逐候选评估，选 cos 最小
        best = None
        for w in cand_words:
            new_sent = self._replace_word_in_sentence(orig_sentence, orig_word, w)
            ids_t = tokenizer(new_sent, return_tensors="pt")["input_ids"].to(input_ids_1xL.device)
            e_t, sparse_t, dense_t = self._get_fused_from_text_ids(image_embeddings_clean, ids_t, image_beit.detach())

            cos = (e_o * e_t).sum(-1) / (e_o.norm(dim=-1) * e_t.norm(dim=-1) + 1e-6)
            score = float(cos.item())  # 越小越好
            if (best is None) or (score < best["score"]):
                best = {
                    "score": score,
                    "sentence": new_sent,
                    "input_ids": ids_t,
                    "sparse": sparse_t,
                    "dense": dense_t,
                    "word": w,
                    "word_pos": word_pos,
                }
        return best

    def cos_sim(self, a, b):
            return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-6)

    def _ssim_global(self, x, y, C1=0.01 ** 2, C2=0.03 ** 2):
        """
        全局 SSIM（不卷积、无滑窗、无 mask）
        x, y: [B,3,H,W] 张量，可以是 0~255 或 0~1
        返回: 标量 1-SSIM（可直接当作 loss）
        """
        # 到 [0,1]，并用 float32 计算更稳
        x = x.float()
        y = y.float()
        if x.max() > 1.5: x = x / 255.0
        if y.max() > 1.5: y = y / 255.0

        # 按通道独立计算全局均值/方差/协方差（在 H、W 上聚合）
        dims = (-2, -1)  # over H,W
        mu_x = x.mean(dim=dims, keepdim=False)  # [B,3]
        mu_y = y.mean(dim=dims, keepdim=False)  # [B,3]
        sigma_x2 = (x * x).mean(dim=dims) - mu_x * mu_x  # [B,3]
        sigma_y2 = (y * y).mean(dim=dims) - mu_y * mu_y  # [B,3]
        sigma_xy = (x * y).mean(dim=dims) - mu_x * mu_y  # [B,3]

        # 按公式计算每个通道的 SSIM，然后对通道与 batch 取平均
        num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)  # [B,3]
        den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2) + 1e-12
        ssim = (num / den).clamp(0, 1)  # [B,3]
        ssim = ssim.mean(dim=1)  # [B]
        loss = 1.0 - ssim  # [B]
        return loss.mean()  # 标量

    def single_prompt_attack_TMM(self, image_path, prompt, tokenizer, mlmtok, mlm):
        self.mlm_tok = mlmtok
        self.mlm_model = mlm
        image_np = self.load_np_image(image_path)
        original_size = image_np.shape[:2]
        image_beit = self.beit3_preprocess(image_np, 224).to(dtype=torch.float16, device='cuda').detach()
        image_sam_unnormed, resize_shape = self.sam_preprocess_1(image_np)
        image_sam_unnormed = image_sam_unnormed.to(dtype=torch.float16, device='cuda')
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device='cuda')
        images = image_sam_unnormed.unsqueeze(0)
        original_clean_image = images.data
        # —— 先跑一次 clean image 的 image_embeddings（后面复用）——
        with torch.no_grad():
            image_embeddings_clean = self.visual_model.image_encoder(self.sam_preprocess_2(images, resize_shape)).detach()
            output = self.mm_extractor.beit3(visual_tokens=image_beit.unsqueeze(0), textual_tokens=input_ids,
                                         text_padding_position=torch.zeros_like(input_ids))
            feat_orig = output["encoder_out"][:, :1, ...]
            feat_orig = self.text_hidden_fcs[0](feat_orig)
            sparse_orig, dense_orig = self.visual_model.prompt_encoder(
                points=None, boxes=None, masks=None, text_embeds=feat_orig,
            )
            sparse_orig = sparse_orig.to(feat_orig.dtype)
            _, _, hs0, attn_flat = self.visual_model.mask_decoder(
                image_embeddings=image_embeddings_clean,
                image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_orig,
                dense_prompt_embeddings=dense_orig,
                multimask_output=False,
                return_hs=True,
                return_attention=True
            )
            z0 = self.fused_repr_from_hs(hs0, self.visual_model.mask_decoder.num_mask_tokens).detach()
            # [1,4096] → [1,1,64,64]
            attn = attn_flat.view(attn_flat.size(0), 1, 64, 64)
            # 归一化到 [0,1]
            attn = attn - attn.amin(dim=(2, 3), keepdim=True)
            attn = attn / (attn.amax(dim=(2, 3), keepdim=True) + 1e-6)
            # 上采样到对抗样本大小（H,W 来自 resize_shape）
            M_up = F.interpolate(attn, size=resize_shape, mode='bilinear', align_corners=False)  # [1,1,H,W]
            M = (M_up >= self.mask_threshold).to(M_up.dtype).detach()
            h, w = resize_shape
            px_image = h * w
            px_crit = M.sum().item()
            px_crit = max(px_crit, 1)
            r = 0.4
            eps_v = self.adv_epsilon  # 你的全局 ε_v（例如 12/255）
            eps_crit = eps_v * (1.0 - r) * (px_image / px_crit)
            eps_ncrit = eps_v * r

            best = self._pick_best_text_replacement(input_ids, tokenizer, image_embeddings_clean, image_beit.detach(), z0)
            if best is not None:
                print(f"[TMM] replace '{prompt}' -> '{best['sentence']}'  (cos={best['score']:.4f})")
                input_ids_adv = best["input_ids"]
                # 固定使用最佳替换词得到的 prompt embeddings
                sparse_embeddings_adv, dense_embeddings_adv = best["sparse"], best["dense"]
            else:
                print("[TMM] no candidate word found; keep original prompt.")
                # 回退：用原文本得到 prompt embeds
                feat = output["encoder_out"][:, :1, ...]
                feat = self.text_hidden_fcs[0](feat)
                sparse_embeddings, dense_embeddings = self.visual_model.prompt_encoder(
                    points=None, boxes=None, masks=None, text_embeds=feat,
                )
                input_ids_adv = input_ids
            _, _, hs_t = self.visual_model.mask_decoder(
                image_embeddings=image_embeddings_clean,
                image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings_adv.detach(),
                dense_prompt_embeddings=dense_embeddings_adv.detach(),
                multimask_output=False,
                return_hs=True,
            )
            z_t = self.fused_repr_from_hs(hs_t, self.visual_model.mask_decoder.num_mask_tokens).detach()
        loss_list = []
        torch.cuda.empty_cache()
        for i in range(self.adv_iters):
            images.requires_grad = True
            image_embeddings = self.visual_model.image_encoder(self.sam_preprocess_2(images, resize_shape))
            _, _, hs_a = self.visual_model.mask_decoder(image_embeddings=image_embeddings,
                                                              image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                                                              sparse_prompt_embeddings=sparse_embeddings_adv,
                                                              dense_prompt_embeddings=dense_embeddings_adv,
                                                              multimask_output=False,
                                                              return_hs =True)
            z_a = self.fused_repr_from_hs(hs_a, self.visual_model.mask_decoder.num_mask_tokens)
            _, _, hs_v = self.visual_model.mask_decoder(image_embeddings=image_embeddings,
                                                                    image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                                                                    sparse_prompt_embeddings=sparse_orig.detach(),
                                                                    dense_prompt_embeddings=dense_orig.detach(),
                                                                    multimask_output=False,
                                                                    return_hs=True)
            z_v = self.fused_repr_from_hs(hs_v, self.visual_model.mask_decoder.num_mask_tokens)
            torch.cuda.empty_cache()
            L_ot = 1 - self.cos_sim(z0, z_t)  # 文本分支（固定）
            L_ov = 1 - self.cos_sim(z0, z_v)  # 图像分支
            L_oa = 1 - self.cos_sim(z0, z_a)  # 联合分支
            L_O = L_ot + L_ov + L_oa
            L_s = self._ssim_global(images, original_clean_image)
            loss = L_O + self.alpha * L_s
            torch.cuda.empty_cache()
            # print('>0 :', (low_res_masks > 0).sum())
            # target_low_res_masks = torch.clamp(low_res_masks.detach(), max=-10)
            # loss = torch.nn.MSELoss()(low_res_masks, target_low_res_masks)
            loss_list.append(loss.detach().item())
            grad = torch.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]
            print(f'mask loss:{loss}')
            # 方向 d_v
            d_v = grad.data.sign()

            # 当前累计扰动（相对原图）
            delta_cur = (images - original_clean_image).detach()

            # 区域步长
            step_crit = self.alpha_crit * d_v
            step_ncrit = self.alpha_ncrit * d_v

            # 只在各自区域内更新
            M_bin = M  # [1,1,H,W]
            delta_next_crit = (delta_cur + step_crit) * M_bin
            delta_next_ncrit = (delta_cur + step_ncrit) * (1.0 - M_bin)

            # 各自做 ℓ∞ 投影到对应预算
            delta_next_crit = torch.clamp(delta_next_crit, min=-eps_crit, max=eps_crit)
            delta_next_ncrit = torch.clamp(delta_next_ncrit, min=-eps_ncrit, max=eps_ncrit)
            delta_new = delta_next_crit + delta_next_ncrit
            images = torch.clamp(original_clean_image + delta_new, min=0, max=255).detach()
            torch.cuda.empty_cache()
        return self.get_cv2_from_torch(images, original_size), loss_list

    def single_prompt_attack_idct(self, image_path, prompt, tokenizer):
        image_np = self.load_np_image(image_path)
        original_size = image_np.shape[:2]
        image_beit = self.beit3_preprocess(image_np, 224).to(dtype=torch.float16, device='cuda')
        image_sam_unnormed, resize_shape = self.sam_preprocess_1(image_np)
        image_sam_unnormed = image_sam_unnormed.to(dtype=torch.float16, device='cuda')
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device='cuda')
        images = image_sam_unnormed.unsqueeze(0)
        original_clean_image = images.data
        clean_image = images.clone().detach()
        output = self.mm_extractor.beit3(visual_tokens=image_beit.unsqueeze(0), textual_tokens=input_ids,
                                         text_padding_position=torch.zeros_like(input_ids))
        feat = output["encoder_out"][:, :1, ...]
        feat = self.text_hidden_fcs[0](feat)
        (
            sparse_embeddings,
            dense_embeddings,
        ) = self.visual_model.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
            text_embeds=feat,
        )
        sparse_embeddings = sparse_embeddings.to(feat.dtype)
        loss_list = []
        clipped_perturbation = 0
        grad = 0
        for i in range(self.adv_iters):
            grad_sum = 0
            for _ in range(self.adv_st_iters):
                st_images = self.spectrum_transform(clean_image.float(), rho=self.adv_rho, sigma=self.adv_sigma).to(clean_image.dtype).detach() + clipped_perturbation
                st_images.requires_grad = True
                image_embeddings = self.visual_model.image_encoder(self.sam_preprocess_2(st_images, resize_shape))
                low_res_masks, _ = self.visual_model.mask_decoder(image_embeddings=image_embeddings,
                                                                  image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                                                                  sparse_prompt_embeddings=sparse_embeddings,
                                                                  dense_prompt_embeddings=dense_embeddings,
                                                                  multimask_output=False)
                print('>0 :', (low_res_masks > 0).sum())
                target_low_res_masks = torch.clamp(low_res_masks.detach(), max=-10)
                loss = torch.nn.MSELoss()(low_res_masks, target_low_res_masks)
                print(f'mask loss:{loss}')
                loss_list.append(loss.detach().item())
                temp_grad = torch.autograd.grad(loss, st_images, retain_graph=False, create_graph=False)[0]
                grad_sum += temp_grad.data
            grad = grad_sum / self.adv_st_iters + grad
            perturbation = self.adv_alpha * grad.data.sign()
            adv_image_unclipped = images.data - perturbation
            clipped_perturbation = torch.clamp(adv_image_unclipped - original_clean_image,
                                               min=-self.adv_epsilon,
                                               max=self.adv_epsilon)
            images = torch.clamp(original_clean_image + clipped_perturbation,
                                 min=0,
                                 max=255).detach()
        return self.get_cv2_from_torch(images, original_size), loss_list

    def resize_back(self, vit_image, new_h, new_w):
        X_scaled = F.interpolate(vit_image, size=(new_h, new_w), mode='bilinear', align_corners=False)
        X_resized = F.interpolate(X_scaled, size=(1024, 1024), mode='bilinear', align_corners=False)
        return X_resized
    def transfer_single_prompt_attack(self, image_path, prompt, tokenizer, lamda=0.5):
        image_np = self.load_np_image(image_path)
        original_size = image_np.shape[:2]
        image_beit = self.beit3_preprocess(image_np, 224).to(dtype=torch.float16, device='cuda')
        image_sam_unnormed, resize_shape = self.sam_preprocess_1(image_np)
        image_sam_unnormed = image_sam_unnormed.to(dtype=torch.float16, device='cuda')
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device='cuda')
        images = image_sam_unnormed.unsqueeze(0)
        original_clean_image = images.data
        output = self.mm_extractor.beit3(visual_tokens=image_beit.unsqueeze(0), textual_tokens=input_ids,
                                         text_padding_position=torch.zeros_like(input_ids))
        feat = output["encoder_out"][:, :1, ...]
        feat = self.text_hidden_fcs[0](feat)
        (
            sparse_embeddings,
            dense_embeddings,
        ) = self.visual_model.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
            text_embeds=feat,
        )
        sparse_embeddings = sparse_embeddings.to(feat.dtype)
        loss_list = []
        for i in range(self.adv_iters):
            s_t = random.uniform(1 - lamda, 1 + lamda)
            new_h = new_w = int(s_t * 1024)
            images.requires_grad = True
            image_embeddings = self.visual_model.image_encoder(self.resize_back(self.sam_preprocess_2(images, resize_shape), new_h, new_w))
            low_res_masks, _ = self.visual_model.mask_decoder(image_embeddings=image_embeddings,
                                                              image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                                                              sparse_prompt_embeddings=sparse_embeddings,
                                                              dense_prompt_embeddings=dense_embeddings,
                                                              multimask_output=False)
            print('>0 :', (low_res_masks > 0).sum())
            target_low_res_masks = torch.clamp(low_res_masks.detach(), max=-10)
            loss = torch.nn.MSELoss()(low_res_masks, target_low_res_masks)
            loss_list.append(loss.detach().item())
            grad = torch.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]
            print(f'mask loss:{loss}')
            perturbation = self.adv_alpha * grad.data.sign()
            adv_image_unclipped = images.data - perturbation
            clipped_perturbation = torch.clamp(adv_image_unclipped - original_clean_image,
                                               min=-self.adv_epsilon,
                                               max=self.adv_epsilon)
            images = torch.clamp(original_clean_image + clipped_perturbation,
                                 min=0,
                                 max=255).detach()
        return self.get_cv2_from_torch(images, original_size), loss_list

    def attack_sam_K(self, image_path, K=400):
        image_np = self.load_np_image(image_path)
        original_size = image_np.shape[:2]
        image_sam_unnormed, resize_shape = self.sam_preprocess_1(image_np)
        image_sam_unnormed = image_sam_unnormed.to(dtype=torch.float16, device='cuda')
        images = image_sam_unnormed.unsqueeze(0)
        original_clean_image = images.data
        sampled_pixel_pos = self.sample_points(original_size, mask=None, sample_size=K)
        points_torch = self.transform_coord(original_size, sampled_pixel_pos, None)
        with torch.no_grad():
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.visual_model.prompt_encoder(
                points=points_torch,
                boxes=None,
                masks=None,
                text_embeds=None,
            )
        loss_list = []
        with trange(self.adv_iters, desc='Attack-SAM-K') as pbar:
            for adv_iter in pbar:
                images.requires_grad = True
                image_embeddings = self.visual_model.image_encoder(self.sam_preprocess_2(images, resize_shape))
                low_res_masks, _ = self.visual_model.mask_decoder(image_embeddings=image_embeddings,
                                                                  image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                                                                  sparse_prompt_embeddings=sparse_embeddings,
                                                                  dense_prompt_embeddings=dense_embeddings,
                                                                  multimask_output=False)
                print('>0 :', (low_res_masks > 0).sum())
                target_low_res_masks = torch.clamp(low_res_masks.detach(), max=-10)
                loss = torch.nn.MSELoss()(low_res_masks, target_low_res_masks)
                loss_list.append(loss.detach().item())
                grad = torch.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]
                print(f'mask loss:{loss}')
                perturbation = self.adv_alpha * grad.data.sign()
                adv_image_unclipped = images.data - perturbation
                clipped_perturbation = torch.clamp(adv_image_unclipped - original_clean_image,
                                                   min=-self.adv_epsilon,
                                                   max=self.adv_epsilon)
                images = torch.clamp(original_clean_image + clipped_perturbation,
                                     min=0,
                                     max=255).detach()
                torch.cuda.empty_cache()
            return self.get_cv2_from_torch(images, original_size), loss_list

    def SRA_attack(self, image_path, interval=50):
        image_np = self.load_np_image(image_path)
        original_size = image_np.shape[:2]
        image_sam_unnormed, resize_shape = self.sam_preprocess_1(image_np)
        image_sam_unnormed = image_sam_unnormed.to(dtype=torch.float16, device='cuda')
        images = image_sam_unnormed.unsqueeze(0)
        original_clean_image = images.data
        sampled_pixel_pos = self.SRA_sample_points(original_size, mask=None, interval=interval)
        points_torch = self.transform_coord(original_size, sampled_pixel_pos, None)
        with torch.no_grad():
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.visual_model.prompt_encoder(
                points=points_torch,
                boxes=None,
                masks=None,
                text_embeds=None,
            )
        loss_list = []
        with trange(self.adv_iters, desc='SRA') as pbar:
            for adv_iter in pbar:
                torch.cuda.empty_cache()
                images.requires_grad = True
                image_embeddings = self.visual_model.image_encoder(self.sam_preprocess_2(images, resize_shape))
                low_res_masks, _ = self.visual_model.mask_decoder(image_embeddings=image_embeddings,
                                                                  image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                                                                  sparse_prompt_embeddings=sparse_embeddings,
                                                                  dense_prompt_embeddings=dense_embeddings,
                                                                  multimask_output=False)
                print('>0 :', (low_res_masks > 0).sum())
                target_low_res_masks = torch.clamp(low_res_masks.detach(), max=-10)
                loss = torch.nn.MSELoss()(low_res_masks, target_low_res_masks)
                loss_list.append(loss.detach().item())
                grad = torch.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]
                print(f'mask loss:{loss}')
                perturbation = self.adv_alpha * grad.data.sign()
                adv_image_unclipped = images.data - perturbation
                clipped_perturbation = torch.clamp(adv_image_unclipped - original_clean_image,
                                                   min=-self.adv_epsilon,
                                                   max=self.adv_epsilon)
                images = torch.clamp(original_clean_image + clipped_perturbation,
                                     min=0,
                                     max=255).detach()
                torch.cuda.empty_cache()
            return self.get_cv2_from_torch(images, original_size), loss_list

    def test(
            self,
            image_np,
            prompts,
            tokenizer,
    ):
        original_size = image_np.shape[:2]
        image_sam_unnormed, resize_shape = self.sam_preprocess_1(image_np)
        image_sam_unnormed = image_sam_unnormed.to(dtype=torch.float16, device='cuda')
        images = image_sam_unnormed.unsqueeze(0)
        image_beit = self.beit3_preprocess(image_np, 224).to(dtype=torch.float16, device='cuda')
        images_evf = image_beit.unsqueeze(0)
        input_ids = [
            tokenizer(prompt, return_tensors="pt").input_ids[0]
            for prompt in prompts
        ]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        input_ids = input_ids.to('cuda')
        attention_masks = input_ids.ne(tokenizer.pad_token_id)
        original_clean_image = image_sam_unnormed.data
        offset = [0, len(prompts)]
        batch_size = 1
        assert batch_size == len(offset) - 1

        images_evf_list = []
        for i in range(len(offset) - 1):
            start_i, end_i = offset[i], offset[i + 1]
            images_evf_i = (
                images_evf[i]
                .unsqueeze(0)
                .expand(end_i - start_i, -1, -1, -1)
                .contiguous()
            )
            images_evf_list.append(images_evf_i)
        images_evf = torch.cat(images_evf_list, dim=0)


        output = self.mm_extractor.beit3(
            visual_tokens=images_evf,
            textual_tokens=input_ids,
            text_padding_position=~attention_masks
        )

        feat = output["encoder_out"][:, :1, ...]

        feat = self.text_hidden_fcs[0](feat)
        feat = torch.split(feat, [offset[i + 1] - offset[i] for i in range(len(offset) - 1)])

        pred_masks = []
        for i in range(len(feat)):
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=feat[i],
            )
            sparse_embeddings = sparse_embeddings.to(feat[i].dtype)

            with torch.no_grad():
                image_embeddings = self.visual_model.image_encoder(self.sam_preprocess_2(images, resize_shape))
            low_res_masks, _ = self.visual_model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            pred_mask = self.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_shape,
                original_size=original_size,
            )
            pred_masks.append(pred_mask[:, 0])

        return pred_masks

    def multi_prompt_attack(
            self,
            image_path,
            prompts,
            tokenizer,
    ):
        image_np = self.load_np_image(image_path)
        original_size = image_np.shape[:2]
        image_sam_unnormed, resize_shape = self.sam_preprocess_1(image_np)
        image_sam_unnormed = image_sam_unnormed.to(dtype=torch.float16, device='cuda')
        images = image_sam_unnormed.unsqueeze(0)
        image_beit = self.beit3_preprocess(image_np, 224).to(dtype=torch.float16, device='cuda')
        images_evf = image_beit.unsqueeze(0)
        input_ids = [
            tokenizer(prompt, return_tensors="pt").input_ids[0]
            for prompt in prompts
        ]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        input_ids = input_ids.to('cuda')
        attention_masks = input_ids.ne(tokenizer.pad_token_id)
        original_clean_image = image_sam_unnormed.data
        offset = [0, len(prompts)]
        batch_size = 1
        assert batch_size == len(offset) - 1

        images_evf_list = []
        for i in range(len(offset) - 1):
            start_i, end_i = offset[i], offset[i + 1]
            images_evf_i = (
                images_evf[i]
                .unsqueeze(0)
                .expand(end_i - start_i, -1, -1, -1)
                .contiguous()
            )
            images_evf_list.append(images_evf_i)
        images_evf = torch.cat(images_evf_list, dim=0)


        output = self.mm_extractor.beit3(
            visual_tokens=images_evf,
            textual_tokens=input_ids,
            text_padding_position=~attention_masks
        )

        feat = output["encoder_out"][:, :1, ...]

        feat = self.text_hidden_fcs[0](feat)
        feat = torch.split(feat, [offset[i + 1] - offset[i] for i in range(len(offset) - 1)])


        for i in range(len(feat)):
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=feat[i],
            )
            sparse_embeddings = sparse_embeddings.to(feat[i].dtype)
            loss_list = []
            for _ in range(self.adv_iters):
                images.requires_grad = True
                image_embeddings = self.visual_model.image_encoder(self.sam_preprocess_2(images, resize_shape))
                low_res_masks, _ = self.visual_model.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                print('>0 :', (low_res_masks > 0).sum())
                target_low_res_masks = torch.clamp(low_res_masks.detach(), max=-10)
                loss = torch.nn.MSELoss()(low_res_masks, target_low_res_masks)
                loss_list.append(loss.detach().item())
                grad = torch.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]
                print(f'mask loss:{loss}')
                perturbation = self.adv_alpha * grad.data.sign()
                adv_image_unclipped = images.data - perturbation
                clipped_perturbation = torch.clamp(adv_image_unclipped - original_clean_image,
                                                   min=-self.adv_epsilon,
                                                   max=self.adv_epsilon)
                images = torch.clamp(original_clean_image + clipped_perturbation,
                                     min=0,
                                     max=255).detach()
                torch.cuda.empty_cache()
        return self.get_cv2_from_torch(images, original_size), loss_list

    def seg_pgd_multi_prompt_attack(
            self,
            image_path,
            prompts,
            tokenizer,
    ):
        image_np = self.load_np_image(image_path)
        original_size = image_np.shape[:2]
        image_sam_unnormed, resize_shape = self.sam_preprocess_1(image_np)
        image_sam_unnormed = image_sam_unnormed.to(dtype=torch.float16, device='cuda')
        images = image_sam_unnormed.unsqueeze(0)
        image_beit = self.beit3_preprocess(image_np, 224).to(dtype=torch.float16, device='cuda')
        images_evf = image_beit.unsqueeze(0)
        input_ids = [
            tokenizer(prompt, return_tensors="pt").input_ids[0]
            for prompt in prompts
        ]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        input_ids = input_ids.to('cuda')
        attention_masks = input_ids.ne(tokenizer.pad_token_id)
        original_clean_image = image_sam_unnormed.data
        offset = [0, len(prompts)]
        batch_size = 1
        assert batch_size == len(offset) - 1

        images_evf_list = []
        for i in range(len(offset) - 1):
            start_i, end_i = offset[i], offset[i + 1]
            images_evf_i = (
                images_evf[i]
                .unsqueeze(0)
                .expand(end_i - start_i, -1, -1, -1)
                .contiguous()
            )
            images_evf_list.append(images_evf_i)
        images_evf = torch.cat(images_evf_list, dim=0)

        output = self.mm_extractor.beit3(
            visual_tokens=images_evf,
            textual_tokens=input_ids,
            text_padding_position=~attention_masks
        )

        feat = output["encoder_out"][:, :1, ...]

        feat = self.text_hidden_fcs[0](feat)
        feat = torch.split(feat, [offset[i + 1] - offset[i] for i in range(len(offset) - 1)])

        for i in range(len(feat)):
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=feat[i],
            )
            sparse_embeddings = sparse_embeddings.to(feat[i].dtype)
            loss_list = []
            criterion = torch.nn.MSELoss()
            for _ in range(self.adv_iters):
                images.requires_grad = True
                image_embeddings = self.visual_model.image_encoder(self.sam_preprocess_2(images, resize_shape))
                low_res_masks, _ = self.visual_model.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                lamb = (i - 1) / (self.adv_iters * 2)
                mask_true = (low_res_masks > 0).int()
                mask_false = (low_res_masks < 0).int()
                print('>0 :', (low_res_masks > 0).sum())
                target_low_res_masks = torch.clamp(low_res_masks.detach(), max=-10)
                loss_t = (1 - lamb) * criterion(mask_true * low_res_masks, target_low_res_masks)
                loss_f = lamb * criterion(mask_false * low_res_masks, target_low_res_masks)
                loss = loss_t + loss_f
                loss_list.append(loss.detach().item())
                grad = torch.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]
                print(f'mask loss:{loss}')
                perturbation = self.adv_alpha * grad.data.sign()
                adv_image_unclipped = images.data - perturbation
                clipped_perturbation = torch.clamp(adv_image_unclipped - original_clean_image,
                                                   min=-self.adv_epsilon,
                                                   max=self.adv_epsilon)
                images = torch.clamp(original_clean_image + clipped_perturbation,
                                     min=0,
                                     max=255).detach()
                torch.cuda.empty_cache()
        return self.get_cv2_from_torch(images, original_size), loss_list

    def cross_prompt_attack(
            self,
            image_path,
            prompts,
            tokenizer,
    ):
        print('cross_prompt_attack!!')
        image_np = self.load_np_image(image_path)
        original_size = image_np.shape[:2]
        image_sam_unnormed, resize_shape = self.sam_preprocess_1(image_np)
        image_sam_unnormed = image_sam_unnormed.to(dtype=torch.float16, device='cuda')
        images = image_sam_unnormed.unsqueeze(0)
        image_beit = self.beit3_preprocess(image_np, 224).to(dtype=torch.float16, device='cuda')
        images_evf = image_beit.unsqueeze(0)
        input_ids = [
            tokenizer(prompt, return_tensors="pt").input_ids[0]
            for prompt in prompts
        ]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        input_ids = input_ids.to('cuda')
        attention_masks = input_ids.ne(tokenizer.pad_token_id)
        original_clean_image = image_sam_unnormed.data
        offset = [0, len(prompts)]
        batch_size = 1
        assert batch_size == len(offset) - 1

        images_evf_list = []
        for i in range(len(offset) - 1):
            start_i, end_i = offset[i], offset[i + 1]
            images_evf_i = (
                images_evf[i]
                .unsqueeze(0)
                .expand(end_i - start_i, -1, -1, -1)
                .contiguous()
            )
            images_evf_list.append(images_evf_i)
        images_evf = torch.cat(images_evf_list, dim=0)

        with torch.no_grad():
            output = self.mm_extractor.beit3(
                visual_tokens=images_evf,
                textual_tokens=input_ids,
                text_padding_position=~attention_masks
            )

            feat = output["encoder_out"][:, :1, ...]

            feat = self.text_hidden_fcs[0](feat)
            feat = torch.split(feat, [offset[i + 1] - offset[i] for i in range(len(offset) - 1)])
            feat = feat[0]
            clean_feat = feat.data

        loss_list = []
        for i in range(self.adv_iters):
            images.requires_grad = True
            feat.requires_grad = True
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=feat,
            )
            sparse_embeddings = sparse_embeddings.to(feat.dtype)
            # image_embeddings = self.visual_model.image_encoder(self.flip(self.swap_patches_in_image_different(self.sam_preprocess_2(images, resize_shape), 16, 4)))
            image_embeddings = self.visual_model.image_encoder(self.sam_preprocess_2(images, resize_shape))
            low_res_masks, _ = self.visual_model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            print('>0 :', (low_res_masks > 0).sum())
            target_low_res_masks = torch.clamp(low_res_masks.detach(), max=-10)
            loss = torch.nn.MSELoss()(low_res_masks, target_low_res_masks)
            loss_list.append(loss.detach().item())
            iters = i + 1
            if i % 15 == 0:
            # if iters % 5 == 0 and iters >= 40 and iters < 95:
                grad = torch.autograd.grad(loss, inputs=[images, feat], retain_graph=False, create_graph=False)
                image_grad = grad[0]
                prompt_grad = grad[1]
                prompt_perturbation = self.adv_alpha_text * prompt_grad.data.sign()
                adv_feat = feat.data + prompt_perturbation
                clipped_feat_perturbation = torch.clamp(adv_feat - clean_feat, min=-self.adv_epsilon_text, max=self.adv_epsilon_text)
                feat = (clean_feat + clipped_feat_perturbation).detach()
            else:
                image_grad = torch.autograd.grad(loss, inputs=images, retain_graph=False, create_graph=False)[0]
            print(f'mask loss:{loss}')
            perturbation = self.adv_alpha * image_grad.data.sign()
            adv_image_unclipped = images.data - perturbation
            clipped_perturbation = torch.clamp(adv_image_unclipped - original_clean_image,
                                               min=-self.adv_epsilon,
                                               max=self.adv_epsilon)
            images = torch.clamp(original_clean_image + clipped_perturbation,
                                 min=0,
                                 max=255).detach()
            torch.cuda.empty_cache()
        # plt.plot(range(1, self.adv_iters + 1), loss_list, marker='o')
        # plt.xlabel('adv iters')
        # plt.ylabel('Loss')
        # plt.title('mask Loss Over iters')
        # plt.grid(True)
        # fig = plt.gcf()  # gcf() 获取当前的 Figure
        # plt.close(fig)  # 关闭当前图表，防止重复显示
        return self.get_cv2_from_torch(images, original_size), loss_list

    def cross_prompt_attack_2(
            self,
            image_path,
            prompts,
            tokenizer,
    ):
        image_np = self.load_np_image(image_path)
        original_size = image_np.shape[:2]
        image_sam_unnormed, resize_shape = self.sam_preprocess_1(image_np)
        image_sam_unnormed = image_sam_unnormed.to(dtype=torch.float16, device='cuda')
        images = image_sam_unnormed.unsqueeze(0)
        image_beit = self.beit3_preprocess(image_np, 224).to(dtype=torch.float16, device='cuda')
        images_evf = image_beit.unsqueeze(0)
        input_ids = [
            tokenizer(prompt, return_tensors="pt").input_ids[0]
            for prompt in prompts
        ]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        input_ids = input_ids.to('cuda')
        text_embeddings = self.mm_extractor.beit3.text_embed(input_ids.detach()).detach()
        clean_text_embeddings = text_embeddings.data
        attention_masks = input_ids.ne(tokenizer.pad_token_id)
        original_clean_image = image_sam_unnormed.data
        offset = [0, len(prompts)]
        batch_size = 1
        assert batch_size == len(offset) - 1

        start_i, end_i = offset[0], offset[1]
        images_evf = (
            images_evf[0]
            .unsqueeze(0)
            .expand(end_i - start_i, -1, -1, -1)
            .contiguous()
        )
        clean_images_evf = images_evf.data

        loss_list = []
        for i in range(self.adv_iters):
            images_evf.requires_grad = True
            images_evf_embeddings = self.mm_extractor.beit3.vision_embed(images_evf, None)
            multiway_split_position = images_evf_embeddings.size(1)

            text_embeddings.requires_grad = True
            mm_embeddings = torch.cat([images_evf_embeddings, text_embeddings], dim=1)
            encoder_padding_mask = torch.cat(
                [
                    torch.zeros(images_evf_embeddings.shape[:-1]).to(images_evf_embeddings.device).bool(),
                    ~attention_masks,
                ],
                dim=1,
            )
            mm_out = self.mm_extractor.beit3.encoder(
                src_tokens=None,
                encoder_padding_mask=encoder_padding_mask,
                attn_mask=None,
                token_embeddings=mm_embeddings,
                multiway_split_position=multiway_split_position,
                incremental_state=None,
                positions=None,
            )
            mm_out["multiway_split_position"] = multiway_split_position
            feat = mm_out["encoder_out"][:, :1, ...]
            feat = self.text_hidden_fcs[0](feat)
            feat = torch.split(feat, [offset[i + 1] - offset[i] for i in range(len(offset) - 1)])
            feat = feat[0]

            images.requires_grad = True
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=feat,
            )
            sparse_embeddings = sparse_embeddings.to(feat.dtype)
            # image_embeddings = self.visual_model.image_encoder(self.flip(self.swap_patches_in_image_different(self.sam_preprocess_2(images, resize_shape), 16, 4)))
            image_embeddings = self.visual_model.image_encoder((self.sam_preprocess_2(images, resize_shape)))
            low_res_masks, _ = self.visual_model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            print('>0 :', (low_res_masks > 0).sum())
            target_low_res_masks = torch.clamp(low_res_masks.detach(), max=-10)
            loss = torch.nn.MSELoss()(low_res_masks, target_low_res_masks)
            loss_list.append(loss.detach().item())
            iters = i + 1
            # if iters % 10 == 0:
            if iters % 5 == 0 and iters >= 40 and iters < 95:
                grad = torch.autograd.grad(loss, inputs=[images, images_evf, text_embeddings], retain_graph=False, create_graph=False)
                image_grad = grad[0]
                images_evf_grad = grad[1]
                text_grad = grad[2]

                prompt_perturbation = self.adv_alpha_text * text_grad.data.sign()
                adv_feat = text_embeddings.data + prompt_perturbation
                clipped_feat_perturbation = torch.clamp(adv_feat - clean_text_embeddings, min=-self.adv_epsilon_text,
                                                        max=self.adv_epsilon_text)
                text_embeddings = (clean_text_embeddings + clipped_feat_perturbation).detach()
            else:
                grad = torch.autograd.grad(loss, inputs=[images, images_evf], retain_graph=False, create_graph=False)
                image_grad = grad[0]
                images_evf_grad = grad[1]
            print(f'mask loss:{loss}')
            image_evf_perturbation = self.adv_alpha_image_evf * images_evf_grad.data.sign()
            adv_evfimage_unclipped = images_evf.data - image_evf_perturbation
            clipped_evfimage_perturbation = torch.clamp(adv_evfimage_unclipped - clean_images_evf,
                                                        min=-self.adv_epsilon_image_evf,
                                                        max=self.adv_epsilon_image_evf)
            images_evf = torch.clamp(clean_images_evf + clipped_evfimage_perturbation,
                                     min=-1,
                                     max=1).detach()
            perturbation = self.adv_alpha * image_grad.data.sign()
            adv_image_unclipped = images.data - perturbation
            clipped_perturbation = torch.clamp(adv_image_unclipped - original_clean_image,
                                               min=-self.adv_epsilon,
                                               max=self.adv_epsilon)
            images = torch.clamp(original_clean_image + clipped_perturbation,
                                 min=0,
                                 max=255).detach()
            torch.cuda.empty_cache()
        # plt.plot(range(1, self.adv_iters + 1), loss_list, marker='o')
        # plt.xlabel('adv iters')
        # plt.ylabel('Loss')
        # plt.title('mask Loss Over iters')
        # plt.grid(True)
        # fig = plt.gcf()  # gcf() 获取当前的 Figure
        # plt.close(fig)  # 关闭当前图表，防止重复显示
        return self.get_cv2_from_torch(images, original_size), loss_list


    def attack(self, images, images_evf, input_ids, resize_list, original_size_list,):
        original_clean_image = images.data
        output = self.mm_extractor.beit3(visual_tokens=images_evf, textual_tokens=input_ids,
                                         text_padding_position=torch.zeros_like(input_ids))
        feat = output["encoder_out"][:, :1, ...]
        feat = self.text_hidden_fcs[0](feat)
        (
            sparse_embeddings,
            dense_embeddings,
        ) = self.visual_model.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
            text_embeds=feat,
        )
        sparse_embeddings = sparse_embeddings.to(feat.dtype)
        target_low_res_masks = torch.full((1, 1, 256, 256), -20, device='cuda', dtype=torch.float16)
        for i in range(self.adv_iters):
            images.requires_grad = True
            image_embeddings = self.visual_model.image_encoder(self.sam_preprocess_norm(images, resize_list[0]))
            low_res_masks, _ = self.visual_model.mask_decoder(image_embeddings=image_embeddings,
                    image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False)
            print('>0 :', (low_res_masks > 0).sum())
            # target_low_res_masks = torch.clamp(low_res_masks.detach(), max=-40)
            loss = torch.nn.MSELoss()(low_res_masks, target_low_res_masks)
            grad = torch.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]
            print(f'mask loss:{loss}')
            perturbation = self.adv_alpha * grad.data.sign()
            adv_image_unclipped = images.data - perturbation
            clipped_perturbation = torch.clamp(adv_image_unclipped - original_clean_image,
                                               min=-self.adv_epsilon,
                                               max=self.adv_epsilon)
            images = torch.clamp(original_clean_image + clipped_perturbation,
                                      min=0,
                                      max=255).detach()
        adv_cv2 = self.get_cv2_from_torch(images.detach(), original_size_list[0])
        return adv_cv2