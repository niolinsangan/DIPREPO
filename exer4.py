import cv2
import numpy as np
import matplotlib.pyplot as plt

I = cv2.imread(r"C:/Users/Niiyyoooww/Desktop/DIP/Nioshiii.jpg")
I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

plt.subplot(4, 2, 1)
plt.imshow(I)
plt.title('Original Image')
plt.axis('off')

g = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
plt.subplot(4, 2, 5)
plt.imshow(g, cmap='gray')
plt.title('Gray Image')
plt.axis('off')

J = cv2.normalize(g, None, alpha=0.3 * 255, beta=0.7 * 255, norm_type=cv2.NORM_MINMAX)
plt.subplot(4, 2, 3)
plt.imshow(J, cmap='gray')
plt.title('Enhanced Image')
plt.axis('off')

D = cv2.normalize(I, None, alpha=0.2 * 255, beta=0.3 * 255, norm_type=cv2.NORM_MINMAX)
D = cv2.normalize(D, None, alpha=0.6 * 255, beta=0.7 * 255, norm_type=cv2.NORM_MINMAX)
plt.subplot(4, 2, 4)
plt.imshow(D)
plt.title('Enhanced Image 2')
plt.axis('off')

plt.subplot(4, 2, 7)
plt.hist(g.ravel(), bins=256, range=[0, 256])
plt.title('Histogram of Gray Image')

m = cv2.equalizeHist(g)
plt.subplot(4, 2, 6)
plt.imshow(m, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

plt.subplot(4, 2, 8)
plt.hist(m.ravel(), bins=256, range=[0, 256])
plt.title('Histogram of Equalized Image')

plt.tight_layout()
plt.show()