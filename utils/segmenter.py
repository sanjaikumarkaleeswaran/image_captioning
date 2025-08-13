import torch
import torchvision
import numpy as np
from PIL import Image, ImageDraw

class Segmenter:
    def __init__(self, score_thresh: float = 0.5):
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
        self.model.eval()
        self.score_thresh = score_thresh

    @torch.no_grad()
    def predict(self, image: Image.Image):
        x = torchvision.transforms.functional.pil_to_tensor(image).float() / 255.0
        out = self.model([x])[0]
        keep = out["scores"] >= self.score_thresh
        return {
            "boxes": out["boxes"][keep].cpu(),
            "labels": out["labels"][keep].cpu(),
            "scores": out["scores"][keep].cpu(),
            "masks": out["masks"][keep].cpu() if "masks" in out else None
        }

def draw_instance_predictions(image: Image.Image, pred: dict) -> Image.Image:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    im = image.convert("RGBA").copy()
    draw = ImageDraw.Draw(im, "RGBA")

    boxes = pred.get("boxes")
    labels = pred.get("labels")
    scores = pred.get("scores")
    masks = pred.get("masks")

    num = len(boxes) if boxes is not None else 0
    for i in range(num):
        box = boxes[i].tolist()
        lbl = int(labels[i].item())
        score = float(scores[i].item())

        color = (50 + (lbl*40)%200, 30 + (lbl*70)%200, 20 + (lbl*90)%200, 120)

        if masks is not None:
            m = masks[i, 0].numpy()
            overlay = Image.new("RGBA", im.size, (0,0,0,0))
            ox, oy = overlay.size
            # sample a subset of mask pixels for speed
            ys, xs = np.where(m > 0.5)
            step = max(1, int(len(xs) / 3000)) if len(xs) else 1
            ov = ImageDraw.Draw(overlay, "RGBA")
            for j in range(0, len(xs), step):
                xj, yj = int(xs[j]), int(ys[j])
                ov.rectangle([xj, yj, xj+1, yj+1], fill=color)
            im = Image.alpha_composite(im, overlay)

        # draw box
        draw.rectangle(box, outline=(color[0], color[1], color[2], 255), width=2)
        draw.text((box[0]+4, box[1]+4), f"class:{lbl} {score:.2f}", fill=(255,255,255,255))

    return im
