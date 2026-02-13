import os, json, numpy as np, torch, evaluate
from dataclasses import dataclass
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    SegformerImageProcessor, SegformerForSemanticSegmentation,
    TrainingArguments, Trainer
)

DATA_ROOT = "mysegdata"
ID2LABEL = json.load(open(os.path.join(DATA_ROOT, "id2label.json")))
ID2LABEL = {int(k): v for k, v in ID2LABEL.items()}
LABEL2ID = {v:k for k,v in ID2LABEL.items()}
NUM_LABELS = len(ID2LABEL)
IGNORE_INDEX = 255
MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"  # good small baseline

class SegDataset(Dataset):
    def __init__(self, split, processor):
        self.img_dir = os.path.join(DATA_ROOT, "images", split)
        self.mask_dir= os.path.join(DATA_ROOT, "masks",  split)
        self.names = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith((".jpg",".jpeg",".png",".webp"))])
        self.processor = processor
    def __len__(self): return len(self.names)
    def __getitem__(self, i):
        name = self.names[i]
        img  = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        mpath= os.path.join(self.mask_dir, name.rsplit(".",1)[0]+".png")
        mask = np.array(Image.open(mpath)).astype("int64")
        enc  = self.processor(images=img, segmentation_maps=mask, return_tensors="pt")
        return {k: v.squeeze(0) for k,v in enc.items()}

# Optional: class weights for imbalance (computed from training masks)
def compute_class_weights():
    counts = np.zeros(NUM_LABELS, dtype=np.int64)
    mdir = os.path.join(DATA_ROOT, "masks", "train")
    for f in os.listdir(mdir):
        m = np.array(Image.open(os.path.join(mdir,f)))
        m = m[m != IGNORE_INDEX]
        hist = np.bincount(m.ravel(), minlength=NUM_LABELS)
        counts += hist
    freq = counts / max(1, counts.sum())
    # Inverse log frequency (common heuristic)
    weights = 1.0 / (np.log(1.02 + freq) + 1e-8)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)

processor = SegformerImageProcessor.from_pretrained(MODEL_NAME, do_reduce_labels=False)
model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    ignore_mismatched_sizes=True,
    id2label=ID2LABEL,
    label2id=LABEL2ID
)

# For tiny datasets, consider freezing the encoder:
# for p in model.segformer.parameters(): p.requires_grad = False

train_ds = SegDataset("train", processor)
val_ds   = SegDataset("val",   processor)

metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = torch.from_numpy(logits)
    # Upsample to labels size
    up = torch.nn.functional.interpolate(
        logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
    )
    pred = up.argmax(dim=1).numpy()
    return metric.compute(
        predictions=pred, references=labels,
        num_labels=NUM_LABELS, ignore_index=IGNORE_INDEX, reduce_labels=False
    )

# (Optional) Weighted loss
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # [B, C, h, w]
        # Upsample to labels
        up = torch.nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        loss = torch.nn.functional.cross_entropy(
            up, labels.long(),
            ignore_index=IGNORE_INDEX,
            weight=self.class_weights.to(up.device) if self.class_weights is not None else None
        )
        return (loss, outputs) if return_outputs else loss

args = TrainingArguments(
    output_dir="segformer_out",
    learning_rate=5e-5,
    num_train_epochs=20,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    fp16=torch.cuda.is_available(),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=50,
    report_to=[],
)

# Choose trainer: weighted or standard
use_weights = True
if use_weights:
    class_w = compute_class_weights()
    trainer = WeightedTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        tokenizer=processor, compute_metrics=compute_metrics,
        class_weights=class_w
    )
else:
    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        tokenizer=processor, compute_metrics=compute_metrics
    )

trainer.train()
trainer.save_model("segformer_custom")
processor.save_pretrained("segformer_custom")