import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import sys
import os

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

json_path = "fig/sa_1.json"  
image_path = "fig/sa_1.jpg"
checkpoint_path = "./weights/tinysam.pth"
model_type = "vit_t"

with open(json_path, 'r') as f:
    data = json.load(f)

# On récupère la première annotation (index 0)
first_ann = data['annotations'][0]
input_point = np.array(first_ann['point_coords'])
# Si le JSON n'a pas de labels explicites, on assume que ce sont des points positifs (1)
input_label = np.array(first_ann.get('point_labels', [1] * len(input_point)))

print(f"Utilisation du prompt ID: {first_ann['id']} | Points: {input_point}")

sys.path.append("..")
from tinysam import sam_model_registry, SamPredictor

device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to(device=device)
predictor = SamPredictor(sam)

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
)

plt.figure(figsize=(10,10))
plt.imshow(image)

# On sélectionne le masque avec le meilleur score de confiance
best_mask_idx = np.argmax(scores)
show_mask(masks[best_mask_idx], plt.gca())
show_points(input_point, input_label, plt.gca())

plt.title(f"Annotation ID: {first_ann['id']} - Score: {scores[best_mask_idx]:.2f}")
plt.axis('off')
plt.savefig("test_json_prompt.png")
plt.show()