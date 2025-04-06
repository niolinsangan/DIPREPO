#DIP exer 5
import cv2
import numpy as np
import matplotlib.pyplot as plt

i = cv2.imread("nioshiii.jpg")
i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)

bitplanes = [((i >> bp) & 1) * 255 for bp in range(8)]


plt.subplot(3,3,1)  # Original Image

plt.imshow(i, cmap="gray")
plt.title("Original Image")
plt.axis("off")

for idx, bp in enumerate(bitplanes):
    plt.subplot(3,3,idx+2)  # Bitplane Images

    plt.imshow(bp, cmap="gray")

    plt.title(f"Bitplane {idx+1}")
    plt.axis("off")
    
plt.show()
