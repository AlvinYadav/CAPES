from PIL import Image
import numpy as np

PALETTE = {
  0:(0,0,0), 7:(190,190,190)  # add other IDs if present
}

m = np.array(Image.open("mysegdata/mask_ids.png"))
color = np.zeros((*m.shape,3), np.uint8)
for cid, rgb in PALETTE.items():
    color[m==cid] = rgb
Image.fromarray(color).save("mask_color.png")

from PIL import Image
import numpy as np

m = np.array(Image.open("mysegdata/mask_ids.png"))
vis = (m * (255 // 7)).astype(np.uint8)  # spread IDs 0..7 across 0..255
Image.fromarray(vis, mode="L").save("mask_stretched.png")

from PIL import Image
import numpy as np
