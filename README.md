![zdfgsdfz.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/jlEXmQDn1tBgBCHjO3ytD.png)

# **Mirage-Photo-Classifier**

> **Mirage-Photo-Classifier** is an image classification vision-language encoder model fine-tuned from **google/siglip2-base-patch16-224** for a binary image authenticity classification task. It is designed to determine whether an image is real or AI-generated (fake) using the **SiglipForImageClassification** architecture.

```py
Classification Report:
              precision    recall  f1-score   support

        Real     0.9781    0.9132    0.9446      5000
        Fake     0.9186    0.9796    0.9481      5000

    accuracy                         0.9464     10000
   macro avg     0.9484    0.9464    0.9463     10000
weighted avg     0.9484    0.9464    0.9463     10000
```

![download.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/FwEjat-T3wv1v1Idiu8Qm.png)

The model categorizes images into two classes:

- **Class 0:** Real  
- **Class 1:** Fake  

---

# **Run with Transformers 🤗**

```python
!pip install -q transformers torch pillow gradio
```

```python
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
```

---

# **Intended Use**

The **Mirage-Photo-Classifier** model is designed to detect whether an image is genuine (photograph) or synthetically generated. Use cases include:

- **AI Image Detection:** Identifying AI-generated images in social media, news, or datasets.  
- **Digital Forensics:** Helping professionals detect image authenticity in investigations.  
- **Platform Moderation:** Assisting content platforms in labeling generated content.  
- **Dataset Validation:** Cleaning and verifying training data for other AI models.
