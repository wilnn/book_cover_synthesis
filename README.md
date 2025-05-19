# Book Cover Synthesis
## stable diffusion XL Model
**The detailed architecture of the UNET, text encoders is shown in the `sdxl_architecture.txt` file in the `src` folder**
- A latent diffusion model created by Stability AI.
- 2 text encoders: OpenCLIP-ViT/G (By OpenAI) and CLIP-ViT/L (By LAION)
- VAE: same autoencoder architecture used for the original Stable Diffusion at a larger batch size (256 vs 9). The encoder downsamples the input image from 1024×1024×3 to 128×128×4
- Unet: three times larger UNet backbone. mainly due to more attention blocks and a larger cross-attention context, as SDXL uses a second text encoder
- Use the scaled linear noise scheduler
- Euler Discrete sampling method, which is faster and more efficient than the original DDPM  
- Epsilon noise prediction type: Predict the added noise during the forward diffusion


### Problem with the base stable diffusion XL model
- Struggle slightly to generate images with text on them<br>
![image](https://github.com/user-attachments/assets/315bac91-68cf-4c4a-891d-620f81c82a3c)<br>
<span style="font-size:5px">Image from their research paper</span>
- limited text prompt of only 77 tokens due to the maximum length of the CLIP encoder. This problem needs to be addressed since the prompts that I use are often at least 200 tokens.
- Even though rare, this model can output a grayscale image even though you don't ask it to. It can be due to the training data of the model has a lot of grayscale images. 
### Problem with fine-tuning the diffusion model
Due to the fact that the epsilon in the noise scheduler is a matrix that has the same shape as the latent image and filled with random number draw from the Gausian distribution no matter the time step, and noise is added with random time steps for each image in the batch at each training step, the model essentially has to learn to predict different noise for the same images at different epochs, which leads to the training loss fluctuating a lot during training. Therefore, I can not use the train loss to monitor during training to see if the model is converging. The good news is that this is completely normal when training stable diffusion models.<br>
![image](https://github.com/user-attachments/assets/359e5a46-0350-410f-8564-16b88e9a0b38)

## Overcome the limitation of the maximum sequence length of the encoder
- The clip text encoder has a maximum sequence length of only 77 tokens. However, the text prompt for this task is often at least 250 tokens. The good news is that the UNET does not limit the sequence length.

- Solution:
  - First, tokenized the entire text prompt using the provided tokenizer that goes with the clip text encoder, and padded, truncated the prompts in the batch to have a max length of 385 tokens.  (The max length size should be divisible by the original max length size(385 is divisible by 77)
  - Then, encode each chunk of 77 tokens using the clip text encoder
  - Finally, concatenate the encoded tokens back together 
  - This big encoded prompt, later on, can be passed directly to the UNET along with the noisy latent
  - The stable diffusion XL model also uses a pooled embedding, which is a token that summarizes the entire prompt. To create the pooled embedding properly for this big prompt, I decided to do the average of the pooled embedding of each 77-token chunk. This ultimately gave me a single token that summarizes this big prompt.

## Training (Fine-tuning):
### LoRA settings:
After several tests and experiments, I have decided that LoRA rank 32 and alpha 32 apply to the K, Q, V projection layers, the in projection layer, and out projection layer in the attention block in UNet and text encoders (also apply to the multi-layer perceptron in the text encoders). I also apply LoRA to the text encoders because I want it to learn the textual information in the prompt during fine-tuning, since the prompts that I used are much longer than what the text encoder was designed for. Applying LoRA to other layers, such as the convolutional layer in the UNET blocks, will cause the model to output grayscale images more often than before fine-tuning.
```Python
unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        #target_modules=["to_k", "to_q", "to_v", "to_out.0", "down_blocks.0.resnets.0.conv1", "down_blocks.0.resnets.1.conv1", "down_blocks.1.resnets.0.conv1", "down_blocks.1.resnets.1.conv1", "down_blocks.2.resnets.0.conv1", "down_blocks.2.resnets.1.conv1", "mid_block.resnets.0.conv1", "mid_block.resnets.1.conv1", "down_blocks.0.resnets.0.conv2", "down_blocks.0.resnets.1.conv2", "down_blocks.1.resnets.0.conv2", "down_blocks.1.resnets.1.conv2", "down_blocks.2.resnets.0.conv2", "down_blocks.2.resnets.1.conv2", "mid_block.resnets.0.conv2", "mid_block.resnets.1.conv2"],# apply the lora layer to all the module/layer that are "to_k", "to_q", "to_v", "to_out.0" in any block that have them (down block, middle block, up block, etc.,)
        target_modules=["to_k", "to_q", "to_v", "to_out.0", 'proj_out', 'proj_in'], # apply the lora layer to all the module/layer that are "to_k", "to_q", "to_v", "to_out.0" in any block that have them (down block, middle block, up block, etc.,)
    )
unet.add_adapter(unet_lora_config)

text_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "mlp.fc1", "mlp.fc2"],
        )
text_encoder_one.add_adapter(text_lora_config)
        text_encoder_two.add_adapter(text_lora_config)
```

### The training (fine-tuning) process:
- Freeze all the parameters in the base model 
- Then, apply the Lora weights to the layers that I want
 
- During training:
  - Start with clean images (1152x896x3) in a batch
  - Pass the images through the VAE encoder and get the latent 
  - Add Gaussian noise with random time steps using the noise scheduler to the latents in the batch
  - random time steps because we want to save training time and computing power
  - Tokenize and encode the text prompts using the text encoders. Each text encoder will encode the prompt (the entire prompt, not half of it for each), and the big prompt that is obtained from each text encoder will be concatenated together to get an even bigger encoded prompt. 
  - Pass the encoded prompts and the noisy image to the UNet
  - The UNet predicts the noise in the noisy latents (not the clean latent itself). This noise is a tensor that has the same size as the input latent. 
  - Compute the loss using MSE between the predicted noise and the epsilon noise added to the latent.
  - Compute gradients, update the parameters, and repeat.

### Training hyperparameters and information:
  - more than 12k training examples
  - 25 epochs with approximately 33k steps, and a batch size of 3
  - Euler Discrete noise scheduler and epsilon noise prediction type, similar to the base model. It is important to use the same noise scheduler and prediction type as the base model because it is how the model was trained to use and work well on. 
  - Constant learning rate scheduler. Due to the noise being added to the image with random time steps, I want to have the same learning rate at each training step so that it can learn to predict different noise equally. 
  - 5% warmup step
  - learning rate for LoRA in UNet: 1e-4. The recommended learning rate for fine-tuning UNet in Stable Diffusion XL using LoRA. 
  - learning rate for LoRA in text encoder 1 (OpenCLIP-ViT/G): 5e-5. This is also the recommended learning rate. This is lower than the one for UNET because we don't want to fine-tune the text encoder too much, as the UNET is the main component in this task that needs to be fine-tuned, not the text encoder. Also, the text encoders are already good enough and just need slight fine-tuning. 
  - learning rate for LoRA in text encoder 2 (CLIP-ViT/L): 1e-5
  - Loss function: MSE between the predicted noise and the actual added noise. 
  - Optimizer: adamw
  - Adam weight decay: 1e-2. This is the default value. 
  - Adam beta 1: 0.9. This is the default value. 
  - Adam beta 2: 0.999. This is the default value.
  - Adam epsilon: 1e-8. This is the default value.
  - gradient clipping with max gradient norm of 1.0. To prevent gradient exploding in the big model like this. 
  - Mixed precision training with fp16 (16-bit floating point). For faster and less resource-intensive training (fine-tuning)

## Inference:
During evaluation and inference:
1. Encode the long prompt using the proposed method
2. Create a random latent image that is the same shape as the latent created by the VAE encoder that filled with random numbers that are normally distributed (Gaussian distribution)(this is called Gaussian noise). In PyTorch, do it by using: `torch.randn(.....)` (if you use a noise scheduler that uses a different kind of noise, that is not Gaussian noise(random number drawn from a Gaussian distribution), then use that kind of noise)
3. Reserve diffusion:
At each time step, the noisy latent and the encoded prompt are passed through the UNET, the UNET predicts the noise at that time step in the image
After predicting the noise at a time step, use the equation of the Euler Discrete sampling method to compute the cleaner latent. 
Repeat this process several times to get a cleaner latent image each time
4. Pass the cleaner latent through the VAE encoder to output the final image

## Run the code
### Setup
- Python version: 3.12.9
- Install dependencies:<br>
  ```bash
  cd book_cover_synthesis
  pip install -r requirements.txt
  accelerate config default
   ```
- Your `default_config.yaml` file at `.cache/huggingface/accelerate/default_config.yaml` should look like this
  
  ```yaml
  {
  "compute_environment": "LOCAL_MACHINE",
  "debug": false,
  "distributed_type": "MULTI_GPU",
  "downcast_bf16": false,
  "enable_cpu_affinity": false,
  "machine_rank": 0,
  "main_training_function": "main",
  "mixed_precision": "no",
  "num_machines": 1,
  "num_processes": 3,
  "rdzv_backend": "static",
  "same_network": false,
  "tpu_use_cluster": false,
  "tpu_use_sudo": false,
  "use_cpu": false,
  #"main_process_port": 29502
  }
  ```
- Change `"num_processes": 3` to the number of processes that you want. This should be the same as the number of GPUs that you want to train the model on.
  For example, if you want to train the model on 1 GPU, then change it to `"num_processes": 1`
  
### Run training
- Make sure that you are in the book_cover_synthesis folder.
- Check what GPUs are available.
- Then do: 
  ```bash
  export CUDA_VISIBLE_DEVICES=0,1,2 
  ```
  `CUDA_VISIBLE_DEVICES` is an environment variable that controls what GPUs the accelerator can use to train the models. `export CUDA_VISIBLE_DEVICES=0,1,2` means
  accelerator can only use 3 GPUs, which are 0, 1, and 2, to train the model. The number of visible GPUs must be greater than or equal to the number of `"num_processes"` that
  you have in the `default_config.yaml` file above. If you only have 1 GPU and `"num_processes"` in the `default_config.yaml` file is 1, then you can just do `export CUDA_VISIBLE_DEVICES=0`
- Get the full path to `book_cover_synthesis/dataset_for_huggingface_filter/train_set`
- To run the fine-tuning script, do:
  ```bash
  export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
  export TRAIN_DIR="the full path that you just get above that takes to book_cover_synthesis/dataset_for_huggingface_filter/train_set"
  accelerate launch src/train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --caption_column="text" \
   --resolution=1024 \
  --train_batch_size=3 \
  --num_train_epochs=25 \
  --checkpointing_steps=3 \
  --lr_scheduler="constant_with_warmup" \
  --learning_rate=1e-4 \
  --mixed_precision="fp16" \
  --seed=42 \
  --output_dir="./models/example_lora" \
  --rank=32 \
  --lr_warmup_steps=950 \
  --train_text_encoder \
  --te1_lr=5e-5 \
  --te2_lr=1e-5
  ```
  An Example for TRAIN_DIR on my computer:<br>
  `export TRAIN_DIR="/home/public/username/project/book_cover_synthesis/dataset_for_huggingface_filter/train_set"`
- If you want to resume training from the latest checkpoint, do:
  ```bash
  accelerate launch src/train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --caption_column="text" \
   --resolution=1024 \
  --train_batch_size=3 \
  --num_train_epochs=25 \
  --checkpointing_steps=3 \
  --lr_scheduler="constant_with_warmup" \
  --learning_rate=1e-4 \
  --mixed_precision="fp16" \
  --seed=42 \
  --output_dir="./models/example_lora" \
  --rank=32 \
  --lr_warmup_steps=950 \
  --train_text_encoder \
  --te1_lr=5e-5 \
  --te2_lr=1e-5 \
  --resume_from_checkpoint="latest"
  ```
  If not done, please also set the environment variables `MODEL_NAME` and `TRAIN_DIR` as above before running the script to resume the training.


### Run validation
- Do:
  ```bash
  python3 src/validation.py \
  --lora_path="./models/example_lora" \
  --log_path="./evaluation_result/clip_scores.txt" \
  --cuda=0 --num_prompts_to_evaluate=6
  ```
  1. `--lora_path` is the path to the folder that was created during training (which should be `./models/example_lora` if you use the training script above.
  This folder should have the file `pytorch_lora_weights.safetensors` inside. Note that if you do not finish the training, then there will be no file named
  `pytorch_lora_weights.safetensors` inside the created folder. You can validate using a model at a checkpoint instead by giving it something like `--lora_path="./models/example_lora/checkpoint-3"` this will use the  `pytorch_lora_weights.safetensors` inside the `checkpoint-3` folder instead.
  2. `-cuda=0` is the GPU that you want to run the validation on. Here, I set it to 0 (the first GPU)
  3. `--num_prompts_to_evaluate=6` is the number of test prompts that you want to use to do the validation. This number should be divisible by 3. Here, I set it to 6.
  4. `--log_path="./evaluation_result/clip_scores.txt"` is the path to save the validation output. You should not change this unless you need to
 
### Run inference
- Do:
  ```bash
  python3 src/inference.py \
  --lora_path="./models/example_lora" \
  --image_save_path="./evaluation_result/images" \
  --cuda=1 \
  --guidance_scale=5
  ```
  1. `--lora_path` and `--cuda` has the same function as above
  2. `--image_save_path` is the path to save the images after generating. You should not change this unless needed to
  3. `--guidance_scale` controls how closely the model will follow the prompt during generation. The lower the number, the freer and more creative the model can be. The values I often use are 5 (default), 7, and 10 

## Results:
### Sample images for human preferences:
- Prompt: "Book Cover - This book title is \"love and mistletoe\". This book publisher is \"beaverstone press llc\". This book genres are romance , contemporary , contemporary romance , holiday , christmas , holiday , cultural , ireland , novella , business , amazon , new adult. an alternate cover edition can be found here . stand alone christmas novella in the ballybeg series of irish romantic comedies . love laughter and a happily ever after during the festive season kissed by christmas true love by new year policeman brian glenn wants a promotion . studying for a degree in criminology is the first step . when a member of ballybeg most notorious family struts into his forensic psychology class his hopes for a peaceful semester vanish . sharon maccarthy is the last woman he should get involved with however hot and bothered she makes him get under his police uniform . can he survive the semester without succumbing to her charms sharon had a rough few months . she knows her future job prospects depend on her finally finishing her degree . when she is paired with her secret crush for the semester project she sees a chance for happiness . can she persuade brian that there is more to her than sequins high heels and a rap sheet." <br>
  ![image](https://github.com/user-attachments/assets/75283a87-a778-4c94-abb2-81e7e5d2e008)

- Prompt: "Book Cover - This book title is \"aru shah and the tree of wishes\". This book publisher is \"rick riordan presents\". This book genres are childrens , middle grade , fantasy , fantasy , mythology , young adult , fiction , adventure , audiobook , childrens , fantasy , magic , fantasy , urban fantasy. war between the devas and the demons is imminent and the otherworld is on high alert . when intelligence from the human world reveals that the sleeper is holding a powerful clairvoyant and her sister captive aru and her friends launch a mission . the captives a pair of twins turn out to be the newest pandava sisters though according to a prophecy one sister is not the celebration of holi the heavenly attendants stage a massage pr rebranding campaign to convince everyone that the pandavas are to be trusted . as much as aru relishes the attention she fears that she is destined to bring destruction to her sisters as the sleeper has predicted . aru believes that the only way to prove her reputation is to find the kalpavriksha the tree that came out of the ocean of milk when it was churned . if she can reach it before the sleeper perhaps she can turn everything around with one what you wish for aru ."<br>
  ![image](https://github.com/user-attachments/assets/8daa717b-30e0-46b1-9ef7-b2c5f12b9c31)

- Prompt: "Book Cover - This book title is "speaker for the dead" . This book publisher is "tor books" . This book Genres tags are science fiction , fiction , fantasy , science fiction fantasy , young adult , audiobook , science fiction , aliens , space , novels , space , space opera . now available in mass market the revised definitive edition of the hugo and nebula classic . in this second book in the saga set years after the terrible war ender wiggin is reviled by history as the xenocide the destroyer of the alien buggers . now ender tells the true story of the war and seeks to stop history from repeating itself . in the aftermath of his terrible war ender wiggin disappeared and a powerful voice arose the speaker for the dead who told the true story of the bugger long years later a second alien race has been discovered but again the aliens ways are strange and frightening again humans die . and it is only the speaker for the dead who is also ender wiggin the xenocide who has the courage to confront the mystery and the for the dead the second novel in orson scott card ender quintet is the winner of the nebula award for best novel and the hugo award for best novel ."<br>
  ![image](https://github.com/user-attachments/assets/a7869965-3f40-47be-b0a6-c57870c5272c)

- Prompt: "Book Cover - This book title is \"love and mistletoe\". This book publisher is \"beaverstone press llc\". This book genres are romance , contemporary , contemporary romance , holiday , christmas , holiday , cultural , ireland , novella , business , amazon , new adult."<br>
  ![image](https://github.com/user-attachments/assets/93beab77-a855-43d9-8be2-e76fdd243610)

- Prompt: "Book Cover - This book title is \"aru shah and the tree of wishes\". This book publisher is \"rick riordan presents\". This book genres are childrens , middle grade , fantasy , fantasy , mythology , young adult , fiction , adventure , audiobook , childrens , fantasy , magic , fantasy , urban fantasy."<br>
  ![image](https://github.com/user-attachments/assets/55931a92-b1c5-4b18-bf8d-1429fd59e4e6)

- Prompt: "Book Cover - This book title is "speaker for the dead" . This book publisher is "tor books" . This book Genres tags are science fiction , fiction , fantasy , science fiction fantasy , young adult , audiobook , science fiction , aliens , space , novels , space , space opera ."<br>
  ![image](https://github.com/user-attachments/assets/3bf0da9f-6134-4e90-bb2a-2a460cda220c)

## ClIP score:
13.6
information about how CLIP scores and how to compute it: https://huggingface.co/docs/diffusers/en/conceptual/evaluation#text-guided-image-generation





