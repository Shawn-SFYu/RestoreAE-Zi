import cv2
import numpy as np
import matplotlib.pyplot as plt


def erosion_dilation(image, min_kernel_size=1, max_kernel_size=5, iterations=1):
    kernel_size = np.random.randint(min_kernel_size, max_kernel_size + 1)

    # Generate a random structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # Perform random erosion or dilation
    operation = np.random.choice(["erosion", "dilation"])
    if operation == "erosion":
        processed_image = cv2.erode(image, kernel, iterations=iterations)
    else:
        processed_image = cv2.dilate(image, kernel, iterations=iterations)

    return processed_image


def rect_painting(image, num_strokes=10, stroke_range=(1, 50), stroke_intensity_range=(-255/5, 255/5)):
    height, width = image.shape
    canvas = np.zeros((height, width), dtype=np.uint8)
    for _ in range(num_strokes):
        # Generate random stroke properties
        stroke_length = np.random.randint(*stroke_range)
        stroke_width = np.random.randint(*stroke_range)
        stroke_intensity = np.random.randint(*stroke_intensity_range)

        # Generate random stroke position
        start_x = np.random.randint(0, width - stroke_length)
        start_y = np.random.randint(0, height - stroke_width)
        end_x = start_x + stroke_length
        end_y = start_y + stroke_width

        # Draw the stroke on the canvas
        canvas[start_y:end_y, start_x:end_x] = stroke_intensity

    # Combine the original image with the painted strokes
    result = cv2.add(image, canvas)

    # Clip values outside the range [0, 255]
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def poly_painting(image, num_strokes, stroke_intensity_range=(0, 255), max_vertices=6):
    height, width = image.shape

    # Generate and draw random painting strokes
    for _ in range(num_strokes):
        # Generate random stroke properties
        stroke_intensity = np.random.randint(*stroke_intensity_range)
        num_vertices = np.random.randint(3, max_vertices + 1)

        # Generate random polygon vertices
        vertices = np.random.randint(0, max(height, width), size=(num_vertices, 2))

        # Create a blank mask for the stroke
        mask = np.zeros((height, width), dtype=np.uint8)

        # Draw the polygon on the mask
        cv2.fillPoly(mask, [vertices], color=stroke_intensity)

        # Combine the stroke mask with the original image
        image = cv2.add(image, mask)

    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def blur(image, min_kernel_size=3, max_kernel_size=10):
    kernel_size = np.random.randint(min_kernel_size, max_kernel_size + 1)
    if not kernel_size % 2:
        kernel_size += 1
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), kernel_size)

    return blurred_image


def add_noise(image, density=0.0005, kernel_size=15):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    noise1 = np.random.choice([0, 255/4], size=image.shape[:2], p=[1 - density, density]).astype(np.float32)
    noise1 = cv2.dilate(noise1, kernel)
    noise2 = np.random.choice([0, 255/4], size=image.shape[:2], p=[1 - density, density]).astype(np.float32)
    noise2 = cv2.dilate(noise2, kernel)

    image = image + noise1 - noise2

    noisy_image = np.clip(image, 0, 255).astype(np.uint8)
    return noisy_image


def identical(image):
    return image


img_processing_pool = [erosion_dilation, rect_painting, blur, add_noise]  # , identical]


if __name__ == "__main__":
    img = cv2.imread("test0.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = ((img.astype(np.float) - np.min(img)) * 255 / (np.max(img) - np.min(img))).astype(np.uint8)

    fig, axs = plt.subplots(3, 2)

    axs[0, 0].imshow(img)
    axs[0, 0].set_title('Original')

    blurred = blur(img)
    axs[0, 1].imshow(blurred)
    axs[0, 1].set_title('blurred')

    rectpaint = rect_painting(img, num_strokes=10)
    axs[1, 0].imshow(rectpaint)
    axs[1, 0].set_title('rectpaint')

    polypaint = poly_painting(img, num_strokes=5)
    axs[1, 1].imshow(polypaint)
    axs[1, 1].set_title('polypaint')

    noise_img = add_noise(img, density=0.001, kernel_size=13)
    axs[2, 0].imshow(noise_img)
    axs[2, 0].set_title('noise_img')

    ed_img = erosion_dilation(img, min_kernel_size=5, max_kernel_size=12, iterations=1)
    axs[2, 1].imshow(ed_img)
    axs[2, 1].set_title('ed_img')

    plt.show()