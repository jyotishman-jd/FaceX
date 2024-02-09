import cv2
import matplotlib.pyplot as plt

# Load an image
image = cv2.imread('images/abc/501.jpg', 0)  # Load as grayscale

# Perform histogram equalization
equalized_image = cv2.equalizeHist(image)

# Display the original and equalized images
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')

plt.tight_layout()
plt.show()
