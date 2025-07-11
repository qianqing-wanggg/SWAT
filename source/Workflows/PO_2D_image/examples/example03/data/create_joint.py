from PIL import Image
import numpy as np

# Define the dimensions
rows, cols = 41, 100
mortar_row = 20

# Create a white image
image_array = np.ones((rows, cols), dtype=np.uint8) * 255

# Set the middle row to black
image_array[mortar_row] = 0

# Convert to PIL image
image = Image.fromarray(image_array, mode='L')

# Save image
output_path = "./joint.png"
image.save(output_path)
