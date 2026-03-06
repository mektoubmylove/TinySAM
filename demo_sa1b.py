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
    
def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

json_path = "fig/sa_1.json"
image_path = "fig/sa_1.jpg"
checkpoint_path = "./weights/tinysam.pth"
model_type = "vit_t"
output_folder = "sa_1img_results"

os.makedirs(output_folder, exist_ok=True)

with open(json_path, 'r') as f:
    data = json.load(f)
annotations = data['annotations']

sys.path.append("..")
from tinysam import sam_model_registry, SamPredictor

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f" Initialisation sur : {device}")
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to(device=device)
predictor = SamPredictor(sam)

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)

score_log_path = os.path.join(output_folder, "summary_scores.txt")

print(f"Traitement de {len(annotations)} annotations...")

with open(score_log_path, "w") as score_file:
    score_file.write("Index | Annotation_ID | Max_Score\n")
    score_file.write("-" * 35 + "\n")

    for i, ann in enumerate(annotations):
        input_point = np.array(ann['point_coords'])
        input_label = np.array(ann.get('point_labels', [1] * len(input_point)))

        # Inférence
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
        )

        # Sélection du meilleur masque
        best_mask_idx = np.argmax(scores)
        max_score = scores[best_mask_idx]

        # Log du score
        score_file.write(f"{i:03d} | {ann['id']:<13} | {max_score:.4f}\n")

        # Visualisation
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(masks[best_mask_idx], plt.gca())
        show_points(input_point, input_label, plt.gca())

        plt.title(f"Ann ID: {ann['id']} - Score: {max_score:.2f}")
        plt.axis('off')

        save_name = os.path.join(output_folder, f"mask_{i:03d}_{ann['id']}.png")
        plt.savefig(save_name, bbox_inches='tight', pad_inches=0.1)
        
        plt.close(fig)

        if (i + 1) % 5 == 0:
            print(f" {i + 1}/{len(annotations)} traités...")

print(f" Images et outputlogs dans : {output_folder}")