# this file is not meant to be run since it was already used
# to format the dataset into the correct format. Running this file again will give error.

import json
import os
from datasets import load_dataset, DatasetDict
import argparse
from tqdm import tqdm
import shutil
import sys

def for_huggingface_in_folder_format(args):
    with open("./dataset_for_huggingface/full_dataset.jsonl", "w") as f:
        text_root_folder = args.text_path
        image_root_folder = args.image_path
        for text_folder in tqdm(os.listdir(text_root_folder)):
            images = set(os.path.splitext(image)[0] for image in os.listdir(os.path.join(image_root_folder, text_folder)))
            
            text_sub_folder = os.path.join(text_root_folder, text_folder)

            for txtfile in tqdm(os.listdir(text_sub_folder)):
                if os.path.splitext(txtfile)[0] not in images:
                    continue
                content = 'Book Cover - '
                #word_count = 0
                with open(os.path.join(text_sub_folder, txtfile), 'r') as file:
                    for idx, line in enumerate(file):
                        if idx == 0:
                            #content += line[5:-3] + '. '
                            content += line[:19] + '"' + line[19:-3] + '". '
                        if idx == 1:
                            content += line[:23] + '"' + line[23:-3] + '". '
                        if idx == 2:
                            continue
                        if idx == 3:
                            continue
                        if idx == 4:
                            #content += "Genres: " + line[27:-3] + '. '
                            content += "This book genres are "+ line[26:-3] + '. '
                        if idx == 5:
                            content += line[:-3] + "."
                print(content)
                sys.exit()
                f.write(json.dumps({'file_name':os.path.splitext(txtfile)[0]+".jpg", 'text':content}) + "\n")
    dataset = load_dataset('json', data_files='./dataset_for_huggingface/full_dataset.jsonl')
    dataset = dataset.shuffle(seed=42)
    split = dataset['train'].train_test_split(test_size=0.08)
    train_dataset = split['train']
    test_dataset = split['test']

    train_dataset.to_json('./dataset_for_huggingface/train_set/metadata.jsonl')
    test_dataset.to_json('./dataset_for_huggingface/test_set/metadata.jsonl')
    
    with open("./dataset_for_huggingface/test_set/metadata.jsonl", "r") as file:
        for line in file:
        # Parse each line as a JSON object
            json_obj = json.loads(line)
            shutil.move("./dataset_for_huggingface/train_set/"+json_obj["file_name"], "./dataset_for_huggingface/test_set")


def normal_jsonl(args):
    with open("./BOOK_DB/full_dataset.jsonl", "w") as f:
        text_root_folder = args.text_path
        image_root_folder = args.image_path
        for text_folder in tqdm(os.listdir(text_root_folder)):
            images = set(os.path.splitext(image)[0] for image in os.listdir(os.path.join(image_root_folder, text_folder)))
            
            text_sub_folder = os.path.join(text_root_folder, text_folder)

            for txtfile in tqdm(os.listdir(text_sub_folder)):
                if os.path.splitext(txtfile)[0] not in images:
                    continue
                content = ''
                #word_count = 0
                with open(os.path.join(text_sub_folder, txtfile), 'r') as file:
                    for idx, line in enumerate(file):
                        if idx == 0:
                            content += line[5:-3] + '. '
                        if idx == 1:
                            content += line[10:-3] + '. '
                        if idx == 2:
                            continue
                        if idx == 3:
                            continue
                        if idx == 4:
                            content += "Genres: " + line[27:-3] + '. '
                        if idx == 5:
                            content += line[:-3] + "."
                f.write(json.dumps({'image':os.path.join(text_sub_folder, os.path.splitext(txtfile)[0]+".jpg"), 'text':content}) + "\n")
    
    dataset = load_dataset('json', data_files='./BOOK_DB/full_dataset.jsonl')
    dataset = dataset.shuffle(seed=42)
    split = dataset['train'].train_test_split(test_size=0.1)
    train_dataset = split['train']
    test_dataset = split['test']

    train_dataset.to_json('./BOOK_DB/train.jsonl')
    test_dataset.to_json('./BOOK_DB/test.jsonl')          


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_path', type=str, default='./BOOK_DB/text')
    parser.add_argument('--image_path', type=str, default='./BOOK_DB/image')
    args = parser.parse_args()
    for_huggingface_in_folder_format(args)
    #normal_jsonl(args)                 
