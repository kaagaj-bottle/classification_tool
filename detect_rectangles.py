
import cv2

# Function to display pixel position and value


def show_pixel_value(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
        # Get the pixel value at (x, y)
        pixel_value = image[y, x]  # OpenCV uses (row, col) => (y, x)
        print(f"Pixel position: ({x}, {y}), Pixel value: {pixel_value}")


# Load the image
image = cv2.imread('sample.png')

# Display the image
cv2.imshow('Image', image)

# Set mouse callback function
cv2.setMouseCallback('Image', show_pixel_value)

# Wait indefinitely until a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
