from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from huggingface_hub import HfApi
import base64
import json
import pickle 
import tensorflow as tf
from datasets import load_dataset
import os.path
from tqdm import tqdm

import re
from typing import List, Tuple


def match_misalignment_label_to_token(
    misalignment_label,
    prompt,
):
  """Matches the misalignment label to the token.

  Args:
    misalignment_label: The misalignment label from RichHF-18K dataset.
    prompt: The prompt from the Pick-a-pic dataset.

  Returns:
    A list of pairs of token and misalignment label.
  """
  delimiters = ',.?!":; '
  pattern = '|'.join(map(re.escape, delimiters))
  # Split by punctuation or space and remove empty tokens.
  tokens = re.split(pattern, prompt)
  tokens = [t for t in tokens if t]

  misalignment_label = misalignment_label.split(' ')
  misalignment_label = [int(l) for l in misalignment_label]
#   assert len(tokens) == len(misalignment_label)
  if len(tokens) != len(misalignment_label):
      print(tokens, "|", misalignment_label)
  labeled_text = ' '.join([f"{t}_{l}" if l == 0 else t for t, l in zip(tokens, misalignment_label)])
  clean_text = ' '.join(tokens)
  return labeled_text, clean_text


def parse_tfrecord(record):
    example = tf.train.Example()
    example.ParseFromString(record.numpy())
    return example

def build_lookup_table(pickapic, split, uids):
    lookup_table = {}
    for item in tqdm(pickapic[split], desc=f"Building lookup table for {split}"):
        if item['image_0_uid'] in uids:
            lookup_table[item['image_0_uid']] = item
        if item['image_1_uid'] in uids:
            lookup_table[item['image_1_uid']] = item
    return lookup_table

def get_caption_and_images(lookup_table, filename):
    uid = filename.split('/')[-1].split('.')[0]
    matching_item = lookup_table[uid]
    caption = matching_item['caption']
    if matching_item['image_0_uid'] == uid:
        image = matching_item['jpg_0']
    elif matching_item['image_1_uid'] == uid:
        image = matching_item['jpg_1']
    results = {
        'uid': uid, 
        'caption': caption, 
        'jpg': image, 
    }
    return results

def get_uids(file_path):
    raw_dataset = tf.data.TFRecordDataset(file_path)
    uids = []
    for raw_record in tqdm(raw_dataset):
        example = parse_tfrecord(raw_record)
        filename = example.features.feature['filename']
        filename = filename.bytes_list.value[0].decode('utf-8')
        uid = filename.split('/')[-1].split('.')[0]
        uids.append(uid)
    return uids

def read_tfrecord_file(file_path, save_root='richhf-18k', split='train', tag=None):
    raw_dataset = tf.data.TFRecordDataset(file_path)
    data_dir_path = os.path.join(save_root, tag)
    lookup_table = lookup_tables[split]

    json_records = {}
    processed_data = list()

    i = 0
    for raw_record in tqdm(raw_dataset):
        if split == 'train':
            folder = f"{i:05d}"
        else:
            folder = f"{i:03d}"
        save_path = os.path.join(data_dir_path, folder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        example = parse_tfrecord(raw_record)
        record = dict()
        record_images = dict()
        for key, value in example.features.feature.items():
            if value.bytes_list.value:
                if key == 'filename':
                    record[key] = value.bytes_list.value[0].decode('utf-8')
                elif key == 'prompt_misalignment_label':
                    token_label = value.bytes_list.value[0].decode()
                    record[key] = token_label
                else:
                    with open(os.path.join(save_path, f'{key}.jpg'), 'wb') as f:
                        f.write(value.bytes_list.value[0])
                    # record[key] = str(key)+'.jpg'
                    record_images[key] = value.bytes_list.value[0]
            elif value.float_list.value:
                record[key] = value.float_list.value[0]
            elif value.int64_list.value:
                record[key] = value.int64_list.value[0]
            else:
                raise RuntimeError

        results_richhf = get_caption_and_images(lookup_table, record['filename'])
        uid = results_richhf['uid']
        with open(os.path.join(save_path, f'{uid}.jpg'), 'wb') as f:
            f.write(results_richhf['jpg'])

        record_images['image'] = results_richhf['jpg']
        record['prompt'] = results_richhf['caption']

        labeled_text, clean_text = match_misalignment_label_to_token(
            record['prompt_misalignment_label'], results_richhf['caption'])
        record['labeled_prompt'] = labeled_text
        record['clean_prompt'] = clean_text

        json_records[uid] = record
        i += 1
        processed_data.append(record | record_images)
    
    with open(os.path.join(save_root, f'{tag}_meta_data.json'), 'w') as json_file:
        json.dump(json_records, json_file, indent=2)

    return processed_data

pickapic = load_dataset("yuvalkirstain/pickapic_v1", num_proc=64, cache_dir='./data')
print("Loaded pickapic dataset")

train_file_path = "./richhf-18k/train.tfrecord"
train_uids = get_uids(train_file_path)
test_file_path = "./richhf-18k/test.tfrecord"
test_uids = get_uids(test_file_path)
dev_file_path = "./richhf-18k/dev.tfrecord"
dev_uids = get_uids(dev_file_path)

lookup_tables = {}
lookup_tables['train'] = build_lookup_table(pickapic, 'train', train_uids + dev_uids)
del pickapic['train']
lookup_tables['test'] = build_lookup_table(pickapic, 'test', test_uids)
del pickapic
print("Built lookup tables")

train_records = read_tfrecord_file(train_file_path, split='train', tag='train')
test_records = read_tfrecord_file(test_file_path, split='test', tag='test')
dev_records = read_tfrecord_file(dev_file_path, split='train', tag='dev')

train_dataset = Dataset.from_list(train_records)
test_dataset = Dataset.from_list(test_records)
dev_dataset = Dataset.from_list(dev_records)

full_dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset,
    "dev": dev_dataset
})

full_dataset.save_to_disk('./data/rich_human_feedback_dataset')