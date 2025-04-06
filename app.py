import gradio as gr
from transformers import AutoImageProcessor
from transformers import SiglipForImageClassification
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Mirage-Photo-Classifier"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Label mapping
labels = {
    "0": "Real",
    "1": "Fake"
}

def classify_image_authenticity(image):
    """Predicts whether the image is real or AI-generated (fake)."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    
    return predictions

# Gradio interface
iface = gr.Interface(
    fn=classify_image_authenticity,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="Mirage Photo Classifier",
    description="Upload an image to determine if it's Real or AI-generated (Fake)."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
