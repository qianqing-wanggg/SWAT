from PIL import Image
import numpy as np

# Redefine the dimensions after code reset
rows, cols = 41, 100
middle_row = 20

# Create a black image
image_array = np.zeros((rows, cols), dtype=np.uint8)

# Parameters based on visual approximation of the uploaded image
triangle_base = 10  # width of one triangle base
triangle_height = 4  # peak height from center
thickness = 1  # thickness of the M-shaped wave

# Create a black background image
image_array = np.ones((rows, cols), dtype=np.uint8)* 255

# Generate M-shaped triangle wave with thickness
for start_col in range(0, cols, triangle_base):
    for i in range(triangle_base):
        col = start_col + i
        if col >= cols:
            break
        if i < triangle_base // 2:
            height = (i * triangle_height) // (triangle_base // 2)
        else:
            height = ((triangle_base - i) * triangle_height) // (triangle_base // 2)

        center_row = middle_row - height
        for t in range(-thickness, thickness + 1):
            row = center_row + t
            if 0 <= row < rows:
                image_array[row, col] = 0  # Apply thickness vertically


# Convert to PIL image
image_sawtooth = Image.fromarray(image_array, mode='L')

# Save image
sawtooth_output_path = "./joint_sawtooth.png"
image_sawtooth.save(sawtooth_output_path)

