import cv2

import numpy as np



def center_spectrum(img):

    f = np.fft.fft2(img)

    fshift = np.fft.fftshift(f)

    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    return magnitude_spectrum



# Load image

image = cv2.imread("nioshiii.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: Image not found or unable to load.")
    exit()




# Calculate centered frequency spectrum

centered_spectrum = center_spectrum(image)



# Display the spectrum (adjust scaling as needed)

# Scale the centered spectrum for better visibility
scaled_spectrum = cv2.normalize(centered_spectrum, None, 0, 255, cv2.NORM_MINMAX)
scaled_spectrum = np.uint8(scaled_spectrum)

cv2.imshow('Centered Spectrum', scaled_spectrum)


cv2.waitKey(0)

cv2.destroyAllWindows()
