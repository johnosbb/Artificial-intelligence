from pathlib import Path

from tokenizers import ByteLevelBPETokenizer


EXPLORE_PARAMETERS=False
OUTPUT_DIRECTORY="./content/KantaiBERT"
INPUT_FILE="./kant.txt"

paths = [str(x) for x in Path(".").glob("**/*.txt")]
# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

print(f"paths = {paths}")
# Hugging Face’s ByteLevelBPETokenizer() will be trained using kant.txt. A BPE tokenizer will 
# break a string or word down into substrings or subwords. There are two main advantages to this, 
# among many others:
#  • The tokenizer can break words into minimal components. Then it will merge these small 
# components into statistically interesting ones. For example, “smaller" and smallest" 
# can become “small,” “er,” and “est.” The tokenizer can go further. We could get “sm" 
# and “all,” for example. In any case, the words are broken down into subword tokens and 
# smaller units of subword parts such as “sm" and “all" instead of simply “small.”
#  • The chunks of strings classified as unknown, unk_token, using WordPiece level encoding, 
# will practically disappear
# Customize training
# Special tokens won't be learned from the data but included directly.

tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

import os
token_dir = OUTPUT_DIRECTORY
if not os.path.exists(token_dir):
  os.makedirs(token_dir)
tokenizer.save_model(token_dir)


from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

tokenizer = ByteLevelBPETokenizer(
    f"{OUTPUT_DIRECTORY}/vocab.json",
    f"{OUTPUT_DIRECTORY}/merges.txt",
)

tokens = tokenizer.encode("The Critique of Pure Reason.").tokens
print(f"Tokens in the sentence: 'The Critique of Pure Reason.': {tokens}")
encoding = tokenizer.encode("The Critique of Pure Reason.")
print(f"Encoding of the sentence: 'The Critique of Pure Reason.': {encoding}")
#  In this model, we will be training the tokenizer with the following parameters:
#  • files=paths is the path to the dataset
#  • vocab_size=52_000 is the size of our tokenizer’s model length
#  • min_frequency=2 is the minimum frequency threshold
#  • special_tokens=[] is a list of special tokens
#  In this case, the list of special tokens is:
#  • <s>: a start token
#  • <pad>: a padding token
#  • </s>: an end token
#  • <unk>: an unknown token
#  • <mask>: the mask token for language modeling

tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

import torch
torch.cuda.is_available()

from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

print(config)

#Re-creating the Tokenizer in Transformers
from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained(OUTPUT_DIRECTORY, max_length=512)

#Initializing a Model From Scratch
from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config=config)
print(model)

print(model.num_parameters())

from datasets import load_dataset
# https://datascience.stackexchange.com/questions/126508/outdated-transformers-textdataset-class-drops-last-block-when-text-overlaps-rep
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py
# Exploring the Parameters
#  The number of parameters is calculated by taking all parameters in the model and adding them 
# up; for example:
#  • The vocabulary (52,000) x dimensions (768)
#  • The size of the vectors is 1 x 768
#  • The many other dimensions found
#  You will note that dmodel = 768. There are 12 heads in the model. The dimension of dk for each head 
# will thus be 𝑑𝑑𝑘𝑘 = 𝑑𝑑𝑚𝑚𝑚𝑚𝑚𝑚𝑚𝑚𝑚𝑚
#  12
#  =64 .
#  This shows, once again, the optimized LEGO® concept of the building blocks of a transformer.
#  We will now see how the number of parameters of a model is calculated and how the figure 
# 84,095,008 is reached.
if EXPLORE_PARAMETERS:
    LP=list(model.parameters())
    lp=len(LP)
    print(lp)
    for p in range(0,lp):
        print(LP[p])


    # Counting the parameters
    np=0
    for p in range(0,lp):#number of tensors
        PL2=True
        try:
            L2=len(LP[p][0]) #check if 2D
        except:
            L2=1             #not 2D but 1D
            PL2=False
        L1=len(LP[p])      
        L3=L1*L2
        np+=L3             # number of parameters per tensor
        if PL2==True:
            print(p,L1,L2,L3)  # displaying the sizes of the parameters
        if PL2==False:
            print(p,L1,L3)  # displaying the sizes of the parameters

        print(np)              # total number of parameters

from transformers import LineByLineTextDataset

#  load the dataset line by line to generate samples for batch training with block_size=128 limiting the length of an example
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=INPUT_FILE,
    block_size=128,
)       


# We need to run a data collator before initializing the trainer. A data collator will take samples 
# from the dataset and collate them into batches. The results are dictionary-like objects.
#  We are preparing a batched sample process for MLM by setting mlm=True.
#  We also set the number of masked tokens to train mlm_probability=0.15. This will determine 
# the percentage of tokens masked during the pretraining process.
#  We now initialize data_collator with our tokenizer, MLM activated, and the proportion of 
# masked tokens set to 0.15:

# Defining a Data Collator
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


#Initializing the Trainer
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir=OUTPUT_DIRECTORY,
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Pre-training the Model
trainer.train()

trainer.save_model(OUTPUT_DIRECTORY)