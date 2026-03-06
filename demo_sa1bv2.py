import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
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
    
def show_points(coords, labels, ax, marker_size=150):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1)   

json_path = "fig/sa_1.json"
image_path = "fig/sa_1.jpg"
checkpoint_path = "./weights/tinysam.pth"
model_type = "vit_t"
output_folder = "tinysam_complete_results"

os.makedirs(output_folder, exist_ok=True)

if not os.path.exists(json_path):
    print(f"Erreur : Le fichier JSON est introuvable à l'adresse {json_path}")
    sys.exit()

with open(json_path, 'r') as f:
    data = json.load(f)
annotations = data['annotations']

sys.path.append("..")
try:
    from tinysam import sam_model_registry, SamPredictor
except ImportError:
    print("Erreur : Assurez-vous que le dossier TinySAM est dans votre PYTHONPATH.")
    sys.exit()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Initialisation de TinySAM sur : {device}")
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to(device=device)
predictor = SamPredictor(sam)

image = cv2.imread(image_path)
if image is None:
    print(f"Erreur : Impossible de charger l'image {image_path}")
    sys.exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w, _ = image_rgb.shape
predictor.set_image(image_rgb)

# Préparation du log
score_log_path = os.path.join(output_folder, "summary_scores.txt")

print(f"Traitement de {len(annotations)} annotations...")

with open(score_log_path, "w") as score_file:
    score_file.write("Index | ID | Max_Score\n")
    score_file.write("-" * 35 + "\n")

    for i, ann in enumerate(annotations):
        input_point = np.array(ann['point_coords'])
        input_label = np.array(ann.get('point_labels', [1] * len(input_point)))

        # Inférence 
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
        )

        # Sélection du meilleur index pour éviter l'erreur de shape (3, H, W)
        best_idx = np.argmax(scores)
        max_score = scores[best_idx]
        
        # Extraction et interpolation des logits du MEILLEUR masque uniquement
        # On passe de (3, 256, 256) à (1, 1, 256, 256) pour l'interpolation
        best_logits_256 = torch.from_numpy(logits[best_idx]).unsqueeze(0).unsqueeze(0)
        
        score_map_tensor = F.interpolate(
            best_logits_256, 
            size=(h, w), 
            mode='bilinear', 
            align_corners=False
        )
        # Transformation en array numpy 2D (H, W)
        score_map = score_map_tensor.squeeze().cpu().numpy()

        # Log du score
        score_file.write(f"{i:03d} | {ann['id']:<8} | {max_score:.4f}\n")

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        # Overlay
        axes[0].imshow(image_rgb)
        show_mask(masks[best_idx], axes[0])
        show_points(input_point, input_label, axes[0])
        axes[0].set_title(f"Overlay (ID: {ann['id']})")
        axes[0].axis('off')

        # Logits (Heatmap)
        im = axes[1].imshow(score_map, cmap='magma')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        axes[1].set_title("Logits (Soft Target)")
        axes[1].axis('off')

        # Masque Binaire
        axes[2].imshow(score_map > 0, cmap='gray')
        axes[2].set_title(f"Binary Mask (Score: {max_score:.2f})")
        axes[2].axis('off')

        # Sauvegarde
        save_path = os.path.join(output_folder, f"result_{i:03d}_{ann['id']}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

        if (i + 1) % 5 == 0:
            print(f"Progression : {i + 1}/{len(annotations)} terminés")

print(f"résultats dans le dossier : {output_folder}")