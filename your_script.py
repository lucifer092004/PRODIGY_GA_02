# Step 1: Install necessary libraries
!pip install torch torchvision torchaudio
!pip install diffusers
!pip install transformers
!pip install accelerate

# Step 2: Import libraries
from diffusers import StableDiffusionPipeline
import torch

# Step 3: Load the model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda")  # Use GPU

# Step 4: Set the prompt and generate image
prompt = "A beautiful sunset over a mountain with a blue gradient sky"
print("Generating image for the prompt:", prompt)
image = pipe(prompt).images[0]

# Step 5: Display the image
image.show()

# Step 6: Save the image
output_path = "/content/generated_image.png"
image.save(output_path)
print(f"Image saved as {output_path}")

# Step 7: (Optional) Download the image to your local machine
from google.colab import files
files.download(output_path)
