import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread(r'nioshiii.jpg', cv2.IMREAD_GRAYSCALE)

img_float = np.float32(img)

f1 = np.fft.fft2(img_float)
f2 = np.fft.fftshift(f1)

plt.subplot(2, 2, 1)
plt.imshow(np.abs(f1), cmap='gray')
plt.title('Frequency Spectrum')

plt.subplot(2, 2, 2)
plt.imshow(np.abs(f2), cmap='gray')
plt.title('Centered Spectrum')
plt.subplot(2, 2, 1)
plt.imshow(np.log(1 + np.abs(f1)), cmap='gray')
plt.title('Frequency Spectrum')
plt.subplot(2, 2, 2)
plt.imshow(np.log(1 + np.abs(f2)), cmap='gray')
plt.title('Centered Spectrum')

f3 = np.log(1 + np.abs(f2))
plt.subplot(2, 2, 3)
plt.imshow(f3, cmap='gray')
plt.title('log(1 + abs(f2))')

img_fft = np.fft.fft2(img_float)
magnitude_spectrum = np.abs(img_fft) 
plt.subplot(2, 2, 4)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum (2D FFT)') 

plt.show()

