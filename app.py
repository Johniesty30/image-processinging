from flask import Flask, render_template, request
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import cv2
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

def grayscale_luminosity(image_array):
    """Converts an RGB image array to grayscale using the luminosity method."""
    grayscale_image = np.dot(image_array[..., :3], [0.21, 0.72, 0.07])
    return grayscale_image


def grayscale_average(image_array):
    """Converts an RGB image array to grayscale using the average method."""
    grayscale_image = np.mean(image_array[..., :3], axis=2)
    return grayscale_image


def gaussian_smoothing(image_array, sigma=1.0):
    """Performs smoothing on the image array using a Gaussian filter."""
    grayscale_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    smoothed_image = gaussian_filter(grayscale_image, sigma=sigma)
    return smoothed_image


def sobel_edge_detection(image_array):
    """Detects edges in the image array using the Sobel operator."""
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        # Convert color image to grayscale
        grayscale_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        grayscale_image = image_array

    sobelx = cv2.Sobel(grayscale_image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(grayscale_image, cv2.CV_64F, 0, 1, ksize=5)
    edges = np.sqrt(sobelx**2 + sobely**2)
    edges = (edges / np.max(edges) * 255).astype(np.uint8)
    return edges

def red_tint(image_array):
    """Adds a red tint to the image."""
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        red_tinted_image = image_array.astype(np.float64)
        red_tinted_image[..., 1:] *= 0.5  # Reduce green and blue channels by half
        return np.clip(red_tinted_image, 0, 255).astype(np.uint8)
    else:
        return np.clip(image_array * 0.5, 0, 255).astype(np.uint8)

def yellow_tint(image_array):
    """Adds a yellow tint to the image."""
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        yellow_tinted_image = image_array.astype(np.float64)
        yellow_tinted_image[..., 2] *= 0.5  # Reduce blue channel by half
        return np.clip(yellow_tinted_image, 0, 255).astype(np.uint8)
    else:
        return np.clip(image_array * 0.5, 0, 255).astype(np.uint8)

def green_tint(image_array):
    """Adds a green tint to the image."""
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        green_tinted_image = image_array.astype(np.float64)
        green_tinted_image[..., 0] *= 0.5  # Reduce red channel by half
        green_tinted_image[..., 2] *= 0.5  # Reduce blue channel by half
        return np.clip(green_tinted_image, 0, 255).astype(np.uint8)
    else:
        return np.clip(image_array * 0.5, 0, 255).astype(np.uint8)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    image_file = request.files['image']
    image = Image.open(image_file)
    image_array = np.array(image)

    operation = request.form['operation']

    if operation == 'grayscale_luminosity':
        processed_image = grayscale_luminosity(image_array)
    elif operation == 'grayscale_average':
        processed_image = grayscale_average(image_array)
    elif operation == 'gaussian_smoothing':
        sigma = float(request.form['sigma'])
        processed_image = gaussian_smoothing(image_array, sigma=sigma)
    elif operation == 'sobel_edge_detection':
        if len(image_array.shape) == 3:
            # Convert color image to grayscale
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        processed_image = sobel_edge_detection(image_array)
    elif operation == 'red_tint':
        processed_image = red_tint(image_array)
    elif operation == 'yellow_tint':
        processed_image = yellow_tint(image_array)
    elif operation == 'green_tint':
        processed_image = green_tint(image_array)
    else:
        return "Invalid operation"

    # Convert processed image array to PIL image for display
    if len(processed_image.shape) == 2:
        # Grayscale image, convert it to RGB for display
        processed_image_pil = Image.fromarray(processed_image.astype('uint8'), 'L').convert('RGB')
    else:
        # Color image, no need to convert
        processed_image_pil = Image.fromarray(processed_image.astype('uint8'))

    # Convert PIL image to base64 string
    buffered = io.BytesIO()
    processed_image_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return render_template('result.html', img_data=img_str)


if __name__ == '__main__':
    app.run(debug=True)
