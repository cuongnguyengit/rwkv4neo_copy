########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, math, random, os, sys
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info
from datasets import load_dataset
from itertools import chain

class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args

        if args.data_type == "symato":
            import sys;
            sys.path.append('../')
            from symato_2944 import Symato
            symato = Symato()
            self.data = symato.tokenize(args.data_file, rev=(args.data_order == "reversed"))
            sample = symato.tids_to_utf8(self.data[-800:])
            print("\n\n- - - [ TRAIN DATA SAMPLE ] - - -\n", sample, "\n\n")
            self.vocab_size = symato.vocab_size()
            self.data_size = len(self.data)
            rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")
            rank_zero_info(f"Data has {self.data_size} samples.")

        elif args.data_type == "chars":
            self.data = open(args.data_file, "r", encoding="utf-8").read()
            rank_zero_info("Building token list...")
            unique = sorted(list(set(self.data)))
            self.vocab_size = len(unique)
            xx = 0
            xxObj = {}
            for u in unique:
                xxObj[xx] = u
                xx += 1
            with open(f"{args.proj_dir}/vocab.json", "w", encoding="utf-8") as vocab_file:
                vocab_file.write(json.dumps(xxObj, ensure_ascii=False))
            self.data_size = len(self.data)
            rank_zero_info(f"Data has {self.data_size} tokens, {self.vocab_size} vocab size.")
            self.stoi = {ch: i for i, ch in enumerate(unique)}
            self.itos = {i: ch for i, ch in enumerate(unique)}

        else:  # unicode
            rank_zero_info("load data...")

            raw_datasets = load_dataset(
                "text",
                data_files=args.data_file,
                cache_dir="./cache/",
            )

            # txt = open(args.data_file, "r", encoding=args.data_type).read()
            # from tokenization_phobert_fast import PhobertTokenizerFast
            os.environ["TOKENIZERS_PARALLELISM"] = "False"
            # tknz = PhobertTokenizerFast("./data/vocab.txt", "./data/bpe.codes", "./data/tokenizer.json")
            # self.vocab_size = 64256  # 251 * 256
            from transformers import AutoTokenizer
            tknz = AutoTokenizer.from_pretrained("/content/drive/MyDrive/llm/checkpoint/gptneo_vietai/")

            def tokenize_function(examples):
                output = tknz(examples['text'])
                return output

            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=8,
                remove_columns=['text'],
                load_from_cache_file=True,
                desc="Running tokenizer on dataset",
            )

            block_size = args.ctx_len + 1

            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
                # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
                total_length = (total_length // block_size) * block_size
                # Split by chunks of max_len.
                result = {
                    k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                    for k, t in concatenated_examples.items()
                }
                # print(len(result["input_ids"]), len(result["input_ids"][0]))
                # result["labels"] = result["input_ids"][:, 1:]
                # result["input_ids"] = result["input_ids"][:, :-1]
                return result

            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=8,
                load_from_cache_file=True,
                desc=f"Grouping texts in chunks of {block_size}",
            )

            self.vocab_size = tknz.vocab_size
            # self.data = tknz.encode(open(args.data_file, "r", encoding=args.data_type).read())
            # self.data_size = len(self.data)

            self.data = lm_datasets['train']
            self.data_size = len(self.data)

            rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")
            rank_zero_info(f"Data has {self.data_size} samples.")

    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    # def __getitem__(self, idx):
    #     args = self.args
    #     ctx_len = args.ctx_len  # ctx_len là độ dài chuỗi token đầu vào
    #     req_len = ctx_len + 1  # cộng thêm một token là kết quả đầu ra
    #     i = np.random.randint(0, self.data_size - req_len)
    #     if args.data_type == "chars":
    #         dix = [self.stoi[s] for s in self.data[i: i + req_len]]
    #     else:
    #         dix = self.data[i: i + req_len]
    #     x = torch.tensor(dix[:-1], dtype=torch.long)
    #     y = torch.tensor(dix[1:], dtype=torch.long)
    #     return x, y

    def __getitem__(self, idx):
        args = self.args
        ctx_len = args.ctx_len  # ctx_len là độ dài chuỗi token đầu vào
        req_len = ctx_len + 1  # cộng thêm một token là kết quả đầu ra

        i = np.random.randint(0, self.data_size)

        x = torch.tensor(self.data[i]['input_ids'][:-1], dtype=torch.long)
        y = torch.tensor(self.data[i]['input_ids'][1:], dtype=torch.long)

        return x, y
