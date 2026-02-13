import os, cv2, torch, numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

MODEL_DIR = "segformer_custom"
processor = SegformerImageProcessor.from_pretrained(MODEL_DIR)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_DIR).eval()

def infer(img_path):
    img = Image.open(img_path).convert("RGB")
    with torch.inference_mode():
        enc = processor(images=img, return_tensors="pt")
        out = model(**enc).logits
        up = torch.nn.functional.interpolate(out, size=img.size[::-1], mode="bilinear", align_corners=False)
        pred = up.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    return np.array(img), pred

def overlay(image_rgb, pred, color=(0,255,255), cls_id=2, alpha=0.45):
    mask = (pred == cls_id).astype(np.uint8)*255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = image_rgb.copy()
    cv2.drawContours(overlay, cnts, -1, color, thickness=cv2.FILLED)
    return cv2.addWeighted(overlay, alpha, image_rgb, 1-alpha, 0)

if __name__ == "__main__":
    img_path = "mysegdata/images/val/img_1010.jpg"
    img, pred = infer(img_path)
    vis = overlay(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), pred, color=(255,200,0), cls_id=2)  # sky
    cv2.imwrite("preview_sky_overlay.jpg", vis)
    print("Saved preview_sky_overlay.jpg")