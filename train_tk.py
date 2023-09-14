from transformers import AutoTokenizer
import os

tokenizer = AutoTokenizer.from_pretrained("../checkpoint/gptneo/", use_fast=True)

print(tokenizer.all_special_tokens)

added_special_tokens = {
    'pad_token': '<|endoftext|>',
    'bos_token': '<|endoftext|>',
    'eos_token': '<|endoftext|>'
}

tokenizer.add_special_tokens(added_special_tokens)

print(tokenizer.all_special_tokens)


def batch_tokenize():
    bz = 1000
    total_count = 0

    dir = '../data/'

    for path in os.listdir(dir):
        print(f"\n{path}")
        path = os.path.join(dir, path)
        tmp = ''
        count = 0
        with open(path, 'r', encoding='utf-8') as rf:
            for line in rf:
                tmp += line
                count += 1
                total_count += 1
                if count % bz == 0:
                    yield tmp
                    tmp = ''
                    print(f"\rCount={count}, total={total_count}, \t", end=' ')
            if tmp:
                yield tmp


new_tokenizer = tokenizer.train_new_from_iterator(text_iterator=batch_tokenize(), vocab_size=25000)

print(new_tokenizer.all_special_tokens)
new_tokenizer.add_special_tokens(added_special_tokens)

print(new_tokenizer.all_special_tokens)

print(tokenizer.tokenize("con đĩ chó này"))
print(new_tokenizer.tokenize("con đĩ chó này"))

tokenizer.save_pretrained("../checkpoint/rwkv4c/")
