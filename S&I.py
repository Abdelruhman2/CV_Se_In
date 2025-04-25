import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ---------- Semantic Segmentation (FCN) ---------- 
def run_fcn(image):
    model = torchvision.models.segmentation.fcn_resnet50(pretrained=True).eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)['out'][0]
    output_predictions = output.argmax(0)

    # Map each class to a color
    colors = np.random.randint(0, 255, size=(21, 3), dtype=np.uint8)
    seg_image = colors[output_predictions.cpu().numpy()]

    return seg_image

# ---------- Instance Segmentation (Mask R-CNN) ---------- 
def run_mask_rcnn(image):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img_tensor = transform(image)

    with torch.no_grad():
        prediction = model([img_tensor])[0]

    masks = prediction['masks']
    labels = prediction['labels']
    scores = prediction['scores']

    mask_img = np.array(image).copy()

    for i in range(len(masks)):
        if scores[i] > 0.8:
            mask = masks[i, 0].mul(255).byte().cpu().numpy()
            color = np.random.randint(0, 255, (1, 3), dtype=int)[0]
            mask_img[mask > 128] = mask_img[mask > 128] * 0.5 + color * 0.5

    return mask_img.astype(np.uint8)

# ---------- Run both models ---------- 
def run_segmentation(image_path):
    # Load image
    orig_img = Image.open(image_path).convert("RGB")

    # Semantic segmentation (FCN)
    sem_seg_img = run_fcn(orig_img)

    # Instance segmentation
    inst_img = run_mask_rcnn(orig_img)

    # Plot results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(orig_img)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Semantic Segmentation (FCN)")
    plt.imshow(sem_seg_img)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Instance Segmentation (Mask R-CNN)")
    plt.imshow(inst_img)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# ---------- Try it ---------- 
# بدل المسار ده بالمسار الصحيح للصورة اللي انت رفعتها
run_segmentation('D:\FCI\Forth_Year\Second_Term\Computer Vision\Sections\SemanticVsInstance\image_test2.jpg')
