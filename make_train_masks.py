# make_train_masks.py
import os, json, cv2, numpy as np
from pathlib import Path
from PIL import Image

import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
import torch.nn.functional as F

# --------- CONFIG ----------
IMAGES_DIR = "mysegdata/images/train"   # input images
OUT_MASKS  = "mysegdata/masks/train"    # output single-channel PNGs
Path(OUT_MASKS).mkdir(parents=True, exist_ok=True)

TARGET_ID2LABEL = {
  0:"background",1:"person",2:"sky",3:"vegetation",4:"animals",5:"water",6:"road",7:"building"
}
LABEL2ID = {v:k for k,v in TARGET_ID2LABEL.items()}

# SAM
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]  # .../image_processing
SAM_TYPE = "vit_h"

SAM_CKPT = PROJECT_ROOT / "src" / "checkpoints" / "sam_vit_h_4b8939.pth"

# Sanity checks
assert SAM_CKPT.is_file(), f"Missing SAM checkpoint at {SAM_CKPT}"
# ViT-H is ~2.6 GB — if it's way smaller, the download is bad (HTML/partial file)
assert SAM_CKPT.stat().st_size > 2 * 1024**3, (
    f"Checkpoint looks too small ({SAM_CKPT.stat().st_size} bytes). "
    "Re-download; you likely saved an HTML page or an incomplete file."
)


# SegFormer on ADE20K (coarse semantics)
SEG_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
# --------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load SAM (AMG)
sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_CKPT).to(device)
amg = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=2000,  # ignore tiny specks
)

# Load SegFormer
processor = AutoImageProcessor.from_pretrained(SEG_MODEL)
segmodel  = AutoModelForSemanticSegmentation.from_pretrained(SEG_MODEL).to(device).eval()
ADE_ID2LABEL = segmodel.config.id2label

# Map ADE label name -> our target class id
def ade_name_to_target_id(name: str) -> int:
    n = name.lower()
    if "person" in n or "people" in n: return LABEL2ID["person"]
    if "animal" in n or "bird" in n or "dog" in n or "cat" in n or "horse" in n: return LABEL2ID["animals"]
    if "sky" in n: return LABEL2ID["sky"]
    if "water" in n or "river" in n or "sea" in n or "lake" in n: return LABEL2ID["water"]
    if "road" in n or "sidewalk" in n or "path" in n or "street" in n or "pavement" in n: return LABEL2ID["road"]
    if "building" in n or "house" in n or "skyscraper" in n or "tower" in n or "bridge" in n: return LABEL2ID["building"]
    if "tree" in n or "plant" in n or "grass" in n or "bush" in n or "vegetation" in n: return LABEL2ID["vegetation"]
    return LABEL2ID["background"]

@torch.inference_mode()
def segformer_target_ids(img_pil: Image.Image) -> np.ndarray:
    H, W = img_pil.size[1], img_pil.size[0]
    enc = processor(images=img_pil, return_tensors="pt").to(device)
    out = segmodel(**enc).logits  # [1,C,h,w]
    up = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
    pred_ids = up.argmax(1)[0].cpu().numpy()
    # convert ADE ids -> our target ids (by name)
    vect = np.vectorize(lambda x: ade_name_to_target_id(ADE_ID2LABEL[int(x)]))
    return vect(pred_ids).astype(np.uint8)  # HxW in {0..7}

def compose_training_mask(image_bgr: np.ndarray, sam_masks: list, coarse_ids: np.ndarray) -> np.ndarray:
    H, W = image_bgr.shape[:2]
    target = coarse_ids.copy()  # fallback fill
    # sort masks by (score * area) so strong, big masks write last
    order = sorted(
        sam_masks, key=lambda m: (float(m.get("predicted_iou",0.0))*m["area"]), reverse=True
    )
    for m in order:
        seg = m["segmentation"].astype(bool)
        if m.get("stability_score", 1.0) < 0.9 or m["area"] < 1000:
            continue
        # majority class inside this region
        vals, counts = np.unique(coarse_ids[seg], return_counts=True)
        maj = int(vals[np.argmax(counts)])
        # optional confidence gate: require majority >= 60%
        if counts.max() / counts.sum() < 0.6:
            continue
        target[seg] = maj
    return target  # uint8 HxW

def process_image(path):
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    # SAM masks
    masks = amg.generate(img)   # list of dicts with 'segmentation', 'area', 'predicted_iou', ...
    # coarse semantics
    coarse = segformer_target_ids(Image.fromarray(img))
    # compose final
    final_ids = compose_training_mask(img, masks, coarse)
    out_png = Path(OUT_MASKS) / (Path(path).stem + ".png")
    Image.fromarray(final_ids, mode="L").save(out_png)

if __name__ == "__main__":
    imgs = [p for p in Path(IMAGES_DIR).glob("*") if p.suffix.lower() in {".jpg",".jpeg",".png",".webp"}]
    for p in imgs:
        process_image(p)
    print(f"Done. Wrote {len(imgs)} masks to {OUT_MASKS}")