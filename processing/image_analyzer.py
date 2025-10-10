import logging
from PIL import Image
from typing import Optional
import torch

logger = logging.getLogger(__name__)


class ImageAnalyzer:
    def __init__(self, vision_model=None, vision_processor=None):
        self.vision_model = vision_model
        self.vision_processor = vision_processor
        logger.info("ImageAnalyzer initialized")

    def set_models(self, vision_model, vision_processor):
        self.vision_model = vision_model
        self.vision_processor = vision_processor
        logger.info("Vision models set for ImageAnalyzer")

    def generate_caption(self, image_path: str, max_length: int = 50) -> str:
        if self.vision_model is None or self.vision_processor is None:
            raise ValueError("Vision model not initialized. Load models first.")

        try:
            logger.info(f"Generating caption for image: {image_path}")

            image = Image.open(image_path).convert('RGB')

            inputs = self.vision_processor(image, return_tensors="pt")

            if torch.cuda.is_available() and next(self.vision_model.parameters()).is_cuda:
                inputs = {k: v.to(self.vision_model.device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = self.vision_model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=3,
                    early_stopping=True
                )

            caption = self.vision_processor.decode(generated_ids[0], skip_special_tokens=True)

            logger.info(f"Generated caption: {caption}")
            return caption

        except Exception as e:
            logger.error(f"Error generating image caption: {str(e)}")
            raise

    def analyze_image(self, image_path: str) -> dict:
        try:
            caption = self.generate_caption(image_path)

            image = Image.open(image_path)
            width, height = image.size
            format_type = image.format

            return {
                "caption": caption,
                "width": width,
                "height": height,
                "format": format_type
            }

        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            raise
