from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

class BLIPCaptioner:
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)

    def generate(self, image: Image.Image, context: str = "", max_length: int = 30) -> str:
        prompt = ("Context: " + context + "\nCaption:") if context else ""
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        out = self.model.generate(**inputs, max_length=max_length)
        return self.processor.decode(out[0], skip_special_tokens=True)
