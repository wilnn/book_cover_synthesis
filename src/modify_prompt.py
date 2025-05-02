# this file is used to quickly modify the prompt in the metadata.jsonl file for the images.

import json

input_path = "/home/public/htnguyen/project/book_cover_synthesis/dataset_for_huggingface_filter/train_set/metadata.jsonl"

output_path = "metadata.jsonl" # New file to save updated content (move this file to
                            #the folder that have the old jsonl file that you want to replace)

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        data = json.loads(line)
        data["text"] = "Book Cover - " + data["text"]
        outfile.write(json.dumps(data) + "\n")

print("New file saved to:", output_path)