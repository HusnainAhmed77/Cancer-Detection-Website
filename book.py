import os
import numpy as np
from PIL import Image

# Create folder if it doesn't exist
os.makedirs("sample_images", exist_ok=True)

# Generate random pixels
img_array = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

# Save as PNG
Image.fromarray(img_array).save("sample_images/test_lesion.png")

print("✅ Sample image saved at sample_images/test_lesion.png")
