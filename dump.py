import cv2
import numpy as np

# Function for Scaling
def scale_image(input_image, scale_factor):
    output_image = cv2.resize(input_image, None, fx=scale_factor, fy=scale_factor)
    return output_image

# Function for Rotation
def rotate_image(input_image, angle):
    rows, cols = input_image.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    output_image = cv2.warpAffine(input_image, M, (cols, rows))
    return output_image

# Function to combine images into a single output
def combine_images(images):
    """
    Combines up to 4 images into a single output image with padding.
    Args:
        images (list): List of images to combine (up to 4)
    Returns:
        numpy.ndarray: Combined image
    """
    if len(images) < 4:
        raise ValueError("At least 4 images are required for combination")
        
    # Find maximum dimensions across all images
    max_height = max(img.shape[0] for img in images)
    max_width = max(img.shape[1] for img in images)
    
    # Add padding between images
    padding = 20
    
    # Create a blank canvas to hold the combined images
    canvas_height = max_height * 2 + padding
    canvas_width = max_width * 2 + padding
    combined_image = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    
    # Resize and place images in the combined canvas with padding
    img0 = cv2.resize(images[0], (max_width, max_height))  # Original
    img1 = cv2.resize(images[1], (max_width, max_height))  # Scaled
    img2 = cv2.resize(images[2], (max_width, max_height))  # Rotated 60
    img3 = cv2.resize(images[3], (max_width, max_height))  # Rotated 45
    
    # Place images in a 2x2 grid with padding
    combined_image[0:max_height, 0:max_width] = img0
    combined_image[0:max_height, max_width + padding:max_width*2 + padding] = img1
    combined_image[max_height + padding:max_height*2 + padding, 0:max_width] = img2
    combined_image[max_height + padding:max_height*2 + padding, max_width + padding:max_width*2 + padding] = img3


    
    return combined_image

# Main execution
if __name__ == "__main__":
    # Load an image
    image = cv2.imread("C:/Users/Niiyyoooww/Desktop/DIP/Nioshiii.jpg", cv2.IMREAD_GRAYSCALE)  # Change to your image path

    # Apply transformations
    scaled_image = scale_image(image, 2)  # Change scale factor as needed
    rotated_image_60 = rotate_image(image, 60)
    rotated_image_45 = rotate_image(image, 45)

    # Combine images into a single output
    combined_image = combine_images([image, scaled_image, rotated_image_60, rotated_image_45])

    # Display the combined result in a resizable window
    cv2.namedWindow('Combined Output', cv2.WINDOW_NORMAL)
    cv2.imshow('Combined Output', combined_image)
    cv2.resizeWindow('Combined Output', 800, 600)  # Set initial window size


    cv2.waitKey(0)
    cv2.destroyAllWindows()
