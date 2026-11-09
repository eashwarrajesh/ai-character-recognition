import gradio as gr
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path

WEIGHTS_PATH = Path(__file__).parent / "model.weights.h5"
if not WEIGHTS_PATH.exists():
    raise FileNotFoundError("model.weights.h5 is missing. Upload the weights file exported from Colab.")

# Recreate the same CNN architecture used for training.
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation="softmax"),
])
model.load_weights(WEIGHTS_PATH)

def distort_image(image, distortion="None", strength=30):
    img = image.copy().astype(np.float32)
    if distortion == "Rotation":
        matrix = cv2.getRotationMatrix2D((14, 14), strength, 1.0)
        img = cv2.warpAffine(img.squeeze(), matrix, (28, 28))[..., np.newaxis]
    elif distortion == "Noise":
        img = np.clip(img + np.random.normal(0, strength / 100, img.shape), 0, 1)
    elif distortion == "Blur":
        kernel = 3 if strength < 50 else 5
        img = cv2.GaussianBlur(img.squeeze(), (kernel, kernel), 0)[..., np.newaxis]
    elif distortion == "Low Contrast":
        img = np.clip((img - 0.5) * (1 - strength / 100) + 0.5, 0, 1)
    return img.astype("float32")

def prepare_image(pil_image):
    img = np.array(pil_image.convert("L"))
    if img.mean() > 127:
        img = 255 - img
    _, binary = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y+h, x:x+w]
    h, w = img.shape
    if h == 0 or w == 0:
        raise gr.Error("The uploaded image appears blank. Upload one clear digit (0–9).")
    scale = 20 / max(h, w)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x0, y0 = (28 - nw) // 2, (28 - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas.astype("float32") / 255.0

def recognize_character(image, distortion, strength):
    if image is None:
        raise gr.Error("Please upload a digit image first.")
    original = prepare_image(image)
    processed = original if distortion == "None" else distort_image(original, distortion, int(strength))
    prediction = model.predict(np.expand_dims(processed, axis=(0, -1)), verbose=0)[0]
    digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(range(10), prediction * 100)
    ax.set_xticks(range(10))
    ax.set_xlabel("Digit")
    ax.set_ylabel("Model score (%)")
    ax.set_title("AI Recognition Confidence")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    probabilities = {str(i): float(prediction[i]) for i in range(10)}
    return ((original * 255).astype("uint8"),
            (processed.squeeze() * 255).astype("uint8"),
            f"Predicted Digit: {digit} | Confidence: {confidence:.2f}%",
            probabilities, fig)

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI Difficult Font and Distorted Character Recognition\n"
                "### Computational Intelligence Project\n"
                "Upload one digit (0–9), apply a distortion, and observe the CNN prediction.")
    with gr.Row():
        input_image = gr.Image(type="pil", label="Upload Character Image")
        with gr.Column():
            distortion = gr.Dropdown(["None", "Rotation", "Noise", "Blur", "Low Contrast"],
                                     value="None", label="Select Distortion")
            strength = gr.Slider(0, 90, value=40, step=5, label="Distortion Strength")
            recognize_btn = gr.Button("Recognize Character", variant="primary")
    with gr.Row():
        original_output = gr.Image(label="Original Character")
        distorted_output = gr.Image(label="Distorted Character")
    prediction_output = gr.Textbox(label="AI Prediction")
    confidence_output = gr.Label(label="Recognition Probabilities", num_top_classes=5)
    graph_output = gr.Plot(label="AI Confidence Graph")
    recognize_btn.click(recognize_character,
                        [input_image, distortion, strength],
                        [original_output, distorted_output, prediction_output,
                         confidence_output, graph_output])

if __name__ == "__main__":
    demo.launch()
