import time
import cv2
import matplotlib.pyplot as plt
from dissection_extractor import run_trajectory_pipeline

IMG = "../left_image_selected/19/119.png"
OUT = ""

_t0 = time.perf_counter()
trajectory = run_trajectory_pipeline(IMG, OUT)
print(f"Pipeline time: {time.perf_counter() - _t0:.3f}s")
print("trajectory shape:", trajectory.shape)
print("trajectory pixels:", int((trajectory > 0).sum()))

# --- display ---
original = cv2.cvtColor(cv2.imread(IMG), cv2.COLOR_BGR2RGB)
overlay = original.copy()
overlay[trajectory > 0] = (0, 255, 0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].imshow(original); axes[0].set_title("Original"); axes[0].axis("off")
axes[1].imshow(overlay);  axes[1].set_title("Overlay");  axes[1].axis("off")
plt.tight_layout()
plt.show()
