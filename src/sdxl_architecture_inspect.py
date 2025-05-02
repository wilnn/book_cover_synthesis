# this file is for inspecting the architecture of the UNet, 
# text encoder 1 and 2 of the sdxl base model, to get the name of the layers to apply LoRA to
# this file is not meant to be run.
# The output of this script is added to the  "sdxl_architecture.txt" file

from diffusers import StableDiffusionXLPipeline
import sys
# Load SDXL pipeline
pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")

# Extract UNet model
unet = pipe.unet

with open("./sdxl_architecture.txt", "w") as file:
    i = 0
    for name, module in unet.named_modules():
        file.write(str(module))
        file.write("\n")
        i+=1
        if i == 1:
            break

    file.write("\n###########################################################################\n")
    file.write("###########################################################################\n")
    file.write("###########################################################################\n\n")

    text_encoder_1 = pipe.text_encoder
    for name, module in text_encoder_1.named_modules():
        file.write(str(module))
        file.write("\n")
        i+=1
        if i == 1:
            break

    file.write("\n###########################################################################")
    file.write("###########################################################################")
    file.write("###########################################################################\n")

    text_encoder_2 = pipe.text_encoder_2

    for name, module in text_encoder_2.named_modules():
        file.write(str(module))
        file.write("\n")
        i+=1
        if i == 1:
            break