# this file is for validating the fine-tuned model

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import torch
import sys
from transformers import AutoTokenizer, PretrainedConfig
import math
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import json
import numpy as np
import argparse

def tokenize_prompt(tokenizer, prompt):
    '''encoded = tokenizer(
        text=prompt,
        padding=True,
        truncation=False,
        return_tensors="pt"
    )
    current_max_len = encoded.input_ids.shape[1]
    padded_length = math.ceil(current_max_len / tokenizer.model_max_length) * tokenizer.model_max_length # tokenizer max length is = to the coressponding text encoder's max length. this is true to the 2 pairs and tokenizer and text encoder and the 2 pairs have same max length
    
    text_inputs = tokenizer(
        text=prompt,
        padding="max_length",
        max_length=padded_length,
        truncation=False,
        return_tensors="pt"
    )'''
    
    padding_length = 539 # this one should be divisible by tokenizer.model_max_length
    text_inputs = tokenizer(
        text=prompt,
        padding="max_length",
        max_length=padding_length,
        truncation=True,
        return_tensors="pt"
    )


    text_input_ids = text_inputs.input_ids
    return text_input_ids

def tokenize_neg_prompt(tokenizer, prompt, max_length):
    
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=False,
        return_tensors="pt"
    )
    '''text_inputs = tokenizer(
        prompt,
        padding="longest",
        #max_length=tokenizer.model_max_length,
        truncation=False,
        return_tensors="pt",
    )'''
    text_input_ids = text_inputs.input_ids
    return text_input_ids

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")

def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None, neg_prompt=False, prompt_max_length=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders): # create a prompt embeds from each text encoders (there are 2 text encoders)
                                                    # each text encoders will encode the same prompt
        
        #print("**********************************************************")
        if tokenizers is not None:
            if neg_prompt:
                assert prompt_max_length is not None
                tokenizer = tokenizers[i]
                text_input_ids = tokenize_neg_prompt(tokenizer, prompt, prompt_max_length)
            else:
                tokenizer = tokenizers[i]
                text_input_ids = tokenize_prompt(tokenizer, prompt)
                
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        count = 0
        concat_embeds = []
        pooled_prompt_embeds = 0
        #print(type(text_input_ids))
        for n in range(0, text_input_ids.shape[-1], max_length):
            

            prompt_embeds = text_encoder(
                text_input_ids[:, n:n+max_length].to(text_encoder.device), output_hidden_states=True, return_dict=False
            )
            '''
            prompt_embeds is a tuple of 3 elements:
            - the first one at index 0 is a torch tensor that is the pooler_output of shape (batch_size, hidden_size)((batch_size, 1280)
                in this case)
            - the second one at index 1 is a torch tensor last_hidden_state (the embeddings of each captions)(the final output from the model) of shape (batch_size, sequence_length, hidden_size) ((batch_size, sequence_length, 1280) in this case)
            - the third one at index 2 is a tuple of torch tensor that are hidden_states. each hidden states in this tuple is an output from a layer of the
                text encoder (the first element in this tuple is the hidden state(embedding) output from the embeddings layer, and second element is the output from the second layer and so on)
                each element(which is hidden state) is of shape (batch_size, sequence_length, hidden_size) ((batch_size, sequence_length, 1280) in this case)(have the same shape as the second element (last_hidden_state) in the big tuple)
            '''

            # We are only ALWAYS interested in the pooled output of the final text encoder
            if count == 0 or count == 1:
                pooled_prompt_embeds = pooled_prompt_embeds + prompt_embeds[0]
            else:
                pooled_prompt_embeds += prompt_embeds[0]

            prompt_embeds = prompt_embeds[-1][-2] # prompt_embeds[-1] return the last element in the tuple(the list of hidden states from each layer)
                                                # then do [-2] return the second to the last layer’s hidden states. get the second to the last layer not the last because in many models, the final layer is too specialized for the training objective of that model(e.g., masked language modeling in BERT).
                                                # The second-to-last layer often contains richer, more general representations, making it useful for downstream tasks like text-to-image alignment.
        
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1) # this line can feel like redundant but is used to make sure that the prompt embeddings will have the shape (batch size, sequence length, hidden size) (for example, maybe the text encoder return the embeddings of shape (batch size, sequence length, 1, hidden size) instead)
            concat_embeds.append(prompt_embeds)
            #print(prompt_embeds.shape)
            count += 1

        pooled_prompt_embeds /= count # averaging the pooled embeddings since each time you the pooled embeddings of a pertion of the prompt so you can add all the pooled embedings of each portion together and take the average. 
        prompt_embeds = torch.cat(concat_embeds, dim=1)
        prompt_embeds_list.append(prompt_embeds)
        #print(len(prompt_embeds_list))

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1) # combine the embeds of the 2 encoders (the 2 encoder encode the same string)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1) # ensure that the pooled embeds have the shape (batch_size, hidden size) because of the same reason 
    return prompt_embeds, pooled_prompt_embeds


def compute_time_ids(original_size, crops_coords_top_left):
    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
    target_size = (1152,896) # this is the original one
    #target_size = (0,0)
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids])
    add_time_ids = add_time_ids.to(device, dtype=torch.float16)
    return add_time_ids

def compute_clip_score(prompts, tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two, pipe, clip_score_fn):
    #print(type(text_encoder_one))

    prompt_embeds, pooled_prompt_embeds = encode_prompt(
                        text_encoders=[text_encoder_one, text_encoder_two],
                        tokenizers=[tokenizer_one, tokenizer_two],
                        prompt=prompts
                        )

    '''
    neg_prompt_embeds, neg_pooled_prompt_embeds = encode_prompt(
                        text_encoders=[text_encoder_one, text_encoder_two],
                        tokenizers=[tokenizer_one, tokenizer_two],
                        prompt=neg_prompt, neg_prompt=True, prompt_max_length=prompt_embeds.shape[1]
                        )'''

    batch = {
        #"original_sizes":[(2048, 1360)],
        "original_sizes":[(350, 425)],
        'crop_top_lefts':[(0, 0)]
    }

    add_time_ids = torch.cat(
    [compute_time_ids(s, c) for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])]
    )


    #unet_added_conditions = {"time_ids": add_time_ids}
    unet_added_conditions = {}

    unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})
    #sys.exit()

    images = pipe(guidance_scale=5,
                num_images_per_prompt=1,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                #negative_prompt_embeds=neg_prompt_embeds,
                #negative_pooled_prompt_embeds=neg_pooled_prompt_embeds,
                height=1152,
                width=896,
                output_type="np").images
    #images[0].save("long_prompt.png")
    #del pipe
    #torch.cuda.empty_cache()       # Optional: clears unused cached memory
    #torch.cuda.ipc_collect()       # Optional: collects inter-process communication memory
    

    def calculate_clip_score(images, prompts):
        images_int = (images * 255).astype("uint8")
        clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
        return round(float(clip_score), 4)

    sd_clip_score = calculate_clip_score(images, prompts)
    return sd_clip_score
    #print(f"CLIP score: {sd_clip_score}")




def evaluate(num_image_to_evaluate, path, args):
    if num_image_to_evaluate % 3 != 0 and num_image_to_evaluate != 0:
        raise Exception("number of image to evaluate should be divisible by 3 and should not be 0")

    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    pipe.load_lora_weights(path)

    '''
    # script for combinig multiple lora
    pipe.load_lora_weights(
        "/home/public/htnguyen/project/diffusers/correctmorelora-rank-128-const1e-4-epsilon-noencoder",
        weight_name="pytorch_lora_weights.safetensors",
        adapter_name="quality_lora"
    )

    pipe.load_lora_weights(
        "/home/public/htnguyen/project/diffusers/lora-linear-epsilon-noencoder",
        weight_name="pytorch_lora_weights.safetensors",
        adapter_name="text_title_lora"
    )

    pipe.set_adapters(["quality_lora", "text_title_lora"], adapter_weights=[0.7, 0.3])
    '''

    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
            model_id,
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
        )
    tokenizer_two = AutoTokenizer.from_pretrained(
            model_id,
            subfolder="tokenizer_2",
            revision=None,
            use_fast=False,
        )

    text_encoder_cls_one = import_model_class_from_model_name_or_path(
            model_id,revision= None
        )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
            model_id,revision= None, subfolder="text_encoder_2"
        )

    text_encoder_one = text_encoder_cls_one.from_pretrained(
            model_id, subfolder="text_encoder", revision=None, variant=None
        ).to(device)
    text_encoder_two = text_encoder_cls_two.from_pretrained(
            model_id, subfolder="text_encoder_2", revision=None, variant=None
        ).to(device)

    clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")


    sum = 0
    # Open the jsonl file and read its content
    with open('./dataset_for_huggingface_filter/test_set/metadata.jsonl', 'r') as file:
        prompts = []
        for i, line in enumerate(file):

            prompts.append(json.loads(line)["text"])
            if i == num_image_to_evaluate:
                break
            if len(prompts) == 3:

                clip_scoree = compute_clip_score(prompts, tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two, pipe, clip_score_fn)
                sum += clip_scoree
                prompts=[]

    return sum/(num_image_to_evaluate//3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    max_length = 77 # text encoder predefined max_length
    '''lora_paths = [
        "/home/public/htnguyen/project/diffusers/rank32-textencoder-const1e-4",
        "/home/public/htnguyen/project/diffusers/final-morelora-rank128-textencoder",
        "/home/public/htnguyen/project/diffusers/final-rank128-textencoder",
        "/home/public/htnguyen/project/diffusers/correctmorelora-rank-128-const1e-4-epsilon-noencoder",
    ]'''
    parser.add_argument('--lora_path', type=str, default=None)
    parser.add_argument('--log_path', type=str, default='./evaluation_result/clip_scores.txt')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--num_prompts_to_evaluate', type=int, default=6)

    args = parser.parse_args()
    device = f"cuda:{str(args.cuda)}"

    lora_paths = [args.lora_path]

    for path in lora_paths:
        avg = evaluate(args.num_prompts_to_evaluate, path, args)
        with open(args.log_path, "a") as file:
            file.write(f"{path}:\n")
            print(f"\n{path}:")
            file.write(f"average clip score: {avg}\n")
            print(f"average clip score: {avg}")
            file.write("************************************\n")
            print("************************************\n")


