# Book Cover Synthesis
## Obtaining the dataset for the grader
The dataset is located on the UMass Lowell gpu2 server: username@cs-gpu2.cs.uml.edu. The path to it is /home/public/htnguyen/project/book_cover_synthesis/dataset_for_huggingface_filter
- First, you need to ssh to username@cs-gpu2.cs.uml.edu
  ```bash
  ssh username@cs-gpu2.cs.uml.edu
  ```
- Then, run these commands:
  ```bash
  cd /home/public
  cp -r /home/public/htnguyen/project/book_cover_synthesis/dataset_for_huggingface_filter .
  ```
  - Now, you have obtained the dataset and copied it to the current folder (at `/home/public`). Move this folder to inside the `book_cover_synthesis` folder that you just cloned or pulled. Please email me with any problems, like permission denied error, etc. 

## Setup
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
## Run training
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
  An Example for TRAIN_DIR on my computer:
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
  ### Training hyperparameters and information:
  - more than 12k training examples
  - 25 epochs with approximately 33k steps, and a batch size of 3
  - Euler Discrete noise scheduler and epsilon noise prediction type, similar to the base model
  - constant learning rate scheduler due to the noise being added to the image with random time steps.  
  - 5% warmup step
  - learning rate for LoRA in UNet: 1e-4
  - learning rate for LoRA in text encoder 1 (OpenCLIP-ViT/G): 5e-5
  - learning rate for LoRA in text encoder 2 (CLIP-ViT/L): 1e-5
  - Loss function: MSE
  - Optimizer: adamw
  - Adam weight decay: 1e-2
  - Adam beta 1: 0.9
  - Adam beta 2: 0.999
  - Adam epsilon: 1e-8
  - gradient clipping with max gradient norm of 1.0
  - Mixed precision training with fp16 (16-bit floating point)

## Run validation
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
 
## Run inference
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







