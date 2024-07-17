import base64
import io

from cerebrium import get_secret
from diffusers import StableDiffusionPipeline

model_name = "runwayml/stable-diffusion-v1-5"
hf_token = get_secret('HF_TOKEN')


def run(prompt: str):
    pipe = StableDiffusionPipeline.from_pretrained(model_name, use_auth_token=hf_token).to("cuda")
    image = pipe(prompt).images[0]
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return {"image": base64.b64encode(buffered.getvalue()).decode("utf-8")}