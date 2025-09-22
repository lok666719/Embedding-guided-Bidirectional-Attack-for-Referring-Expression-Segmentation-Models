# Proxy-Embedding as an Adversarial Teacher: An Embedding-Guided Bidirectional Attack for Referring Expression Segmentation Models

## Description
We propose **Proxy-Embedding as an Adversarial Teacher (PEAT)** — an embedding-guided bidirectional adversarial attack framework for Referring Expression Segmentation (RES) models. The method crafts cross-text transferable adversarial examples by leveraging proxy embeddings as an adversarial teacher. The paper is currently under submission; detailed model diagrams and extended results will be released after acceptance.

## Datasets
We follow dataset preparation and conventions from EVF-SAM. Please obtain datasets and follow their usage rules from the original projects.

- **RefCOCO / RefCOCO+ / RefCOCOg**  
  Download / prepare following EVF-SAM instructions: https://github.com/hustvl/evf-sam?tab=readme-ov-file#-early-vision-language-fusion-for-text-prompted-segment-anything-model-

- **Baselines / Related repos**  
  - EVF-SAM, EVF-SAM2: https://github.com/hustvl/evf-sam?tab=readme-ov-file#-early-vision-language-fusion-for-text-prompted-segment-anything-model-  
  - DMMI: https://github.com/toggle1995/RIS-DMMI/tree/main  
  - LAVT: https://github.com/yz93/LAVT-RIS/tree/main?tab=readme-ov-file

> Note: Some datasets or model weights may require permission or registration from their original authors — follow the instructions in those repositories.

## Requirement
Recommended: Linux, NVIDIA GPU, CUDA 11.7, conda/python 3.9.  
Main dependencies (tested):
python==3.9\
torch==2.0.1+cu117\
torchvision==0.15.2+cu117\
torchaudio==2.0.2\
transformers==4.45.2\
accelerate==1.0.1\
deepspeed==0.15.3\
bitsandbytes==0.41.1\
timm==0.4.12\
opencv-python==4.10.0.84\
pillow==9.4.0\
numpy==1.23.2\
scipy==1.11.2\
matplotlib==3.9.2\
pandas==2.2.3\
safetensors\
pycocotools\
einops\
ftfy\
tqdm
## Implementation
Just download this repository and open it using PyCharm (or your preferred IDE).  

## Attack and Evaluate

Generate adversarial examples and evaluate:

```bash
python adv_eval.py --ckpt checkpoints/your_model.pth --data-root path/to/refcoco/ --output results/adv_examples/
python adv_eval_test.py --ckpt checkpoints/your_model.pth --data-root path/to/refcoco/ --output results/eval/
```
