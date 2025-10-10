import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    BitsAndBytesConfig,
    pipeline
)
from sentence_transformers import SentenceTransformer
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self):
        self.llm_model = None
        self.llm_tokenizer = None
        self.embedding_model = None
        self.vision_model = None
        self.vision_processor = None
        self.whisper_pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing ModelLoader with device: {self.device}")

    def load_llm(self, model_name: str = "Qwen/Qwen3-4B-Instruct-2507", use_4bit: bool = True):
        try:
            logger.info(f"Loading LLM model: {model_name}")

            if use_4bit and self.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )

                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="cuda",
                    trust_remote_code=True
                )
            else:
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
                if self.device == "cpu":
                    self.llm_model = self.llm_model.to(self.device)

            self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

            logger.info("LLM model loaded successfully")
            return self.llm_model, self.llm_tokenizer

        except Exception as e:
            logger.error(f"Error loading LLM model: {str(e)}")
            raise

    def load_embedding_model(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            logger.info(f"Loading embedding model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)

            if self.device == "cuda":
                self.embedding_model = self.embedding_model.to(self.device)

            logger.info("Embedding model loaded successfully")
            return self.embedding_model

        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise

    def load_vision_model(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        try:
            logger.info(f"Loading vision model: {model_name}")
            from transformers import BlipProcessor, BlipForConditionalGeneration

            self.vision_processor = BlipProcessor.from_pretrained(model_name)
            self.vision_model = BlipForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )

            if self.device == "cuda":
                self.vision_model = self.vision_model.to(self.device)

            logger.info("Vision model loaded successfully")
            return self.vision_model, self.vision_processor

        except Exception as e:
            logger.error(f"Error loading vision model: {str(e)}")
            raise

    def load_whisper_model(self, model_name: str = "openai/whisper-base"):
        try:
            logger.info(f"Loading Whisper model: {model_name}")

            self.whisper_pipeline = pipeline(
                "automatic-speech-recognition",
                model=model_name,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )

            logger.info("Whisper model loaded successfully")
            return self.whisper_pipeline

        except Exception as e:
            logger.error(f"Error loading Whisper model: {str(e)}")
            raise

    def generate_text(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        if self.llm_model is None or self.llm_tokenizer is None:
            raise ValueError("LLM model not loaded. Call load_llm() first.")

        try:
            # Qwen models require chat formatting
            formatted_prompt = self.llm_tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.llm_tokenizer(formatted_prompt, return_tensors="pt").to(self.llm_model.device)

            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=self.llm_tokenizer.pad_token_id,
                    eos_token_id=self.llm_tokenizer.eos_token_id
                )

            generated_text = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text

        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise


    def generate_embeddings(self, texts: list) -> list:
        if self.embedding_model is None:
            raise ValueError("Embedding model not loaded. Call load_embedding_model() first.")

        try:
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()

        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def cleanup(self):
        logger.info("Cleaning up models from memory")
        if self.llm_model is not None:
            del self.llm_model
            self.llm_model = None
        if self.embedding_model is not None:
            del self.embedding_model
            self.embedding_model = None
        if self.vision_model is not None:
            del self.vision_model
            self.vision_model = None
        if self.whisper_pipeline is not None:
            del self.whisper_pipeline
            self.whisper_pipeline = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Model cleanup completed")


model_loader_instance = ModelLoader()
