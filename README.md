# Book Cover Synthesis
## Obtaining the dataset for the grader
The dataset is located in the UMass Lowell gpu2 server: username@cs-gpu2.cs.uml.edu. The path to it is /home/public/htnguyen/project/book_cover_synthesis/dataset_for_huggingface_filter
- First you need to ssh to username@cs-gpu2.cs.uml.edu
  ```bash
  ssh username@cs-gpu2.cs.uml.edu
  ```
- Then, run these commands:
  ```bash
  cd /home/public
  cp -r /home/public/htnguyen/project/book_cover_synthesis/dataset_for_huggingface_filter .
  ```
  - Now, you have obtained the dataset and copied it to the current folder (at `/home/public`). Move this folder to inside the `Book_Cover_synthesis` that you just cloned or pulled. Please email me with any problems, like permission denied error, etc. 

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
  3. `--guidance_scale` controls how closely the model will follow the prompt during generating. The lower the number, the freer and more creative the model can be. The values I often use are 5 (default), 7, and 10 




