import cv2
import numpy as np
import matplotlib.pyplot as plt

def scale_image(image, scale_factor, method):
    return cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=method)

def rotate_image(image, angle):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    return cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]))

def display_image(image, title, position, cmap=None, xticks=None, yticks=None):
    plt.subplot(2, 2, position)
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('on')
    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)

def main():
    image_path = r"C:/Users/Niiyyoooww/Desktop/DIP/Nioshiii.jpg"
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Unable to load the image. Check the file path.")
        exit()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 8))

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    display_image(gray_image, 'Original Image (Gray)', 1, cmap='gray', xticks=[200, 400, 600], yticks=[200, 400, 600])

    scaling_factor = 5  # Fixed scaling factor
    scaled_image = scale_image(image, scaling_factor, cv2.INTER_LINEAR)
    gray_scaled_image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY)
    display_image(gray_scaled_image, 'Scaled Image (Gray)', 2, cmap='gray', xticks=[1000, 2000, 3000], yticks=[500, 1000, 1500, 2000])

    rotated_image_60 = rotate_image(scaled_image, 60)
    gray_rotated_image_60 = cv2.cvtColor(rotated_image_60, cv2.COLOR_BGR2GRAY)
    display_image(gray_rotated_image_60, 'Rotated Image 60deg (Gray)', 3, cmap='gray')

    rotated_image_45 = rotate_image(scaled_image, 45)
    gray_rotated_image_45 = cv2.cvtColor(rotated_image_45, cv2.COLOR_BGR2GRAY)
    display_image(gray_rotated_image_45, 'Rotated Image 45deg (Gray)', 4, cmap='gray')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 8))

    display_image(image, 'Original Image', 1, xticks=[100, 200, 300, 400], yticks=[100, 200, 300, 400])

    bilinear_image = scale_image(image, 5, cv2.INTER_LINEAR)
    display_image(bilinear_image, 'Bilinear Image', 2, xticks=[500, 1000, 1500, 2000, 2500], yticks=[500, 1000, 1500, 2000])

    nearest_image = scale_image(image, 5, cv2.INTER_NEAREST)
    display_image(nearest_image, 'Nearest Image', 3, xticks=[500, 1000, 1500, 2000, 2500], yticks=[500, 1000, 1500, 2000])

    bicubic_image = scale_image(image, 5, cv2.INTER_CUBIC)
    display_image(bicubic_image, 'Bicubic Image', 4, xticks=[500, 1000, 1500, 2000, 2500], yticks=[500, 1000, 1500, 2000])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()