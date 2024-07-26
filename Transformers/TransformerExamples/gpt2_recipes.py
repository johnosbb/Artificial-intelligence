import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import random
import datetime
import time

#https://github.com/tuangatech/01-Recipes-Generation-GPT2/blob/main/Recipes_Generation.ipynb
#https://tuanatran.medium.com/fine-tuning-large-language-model-with-hugging-face-pytorch-adce80dce2ad

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device {device}")

model_name = "gpt2-medium"  # options: ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
model_save_path = './models'

configuration = GPT2Config.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name, config=configuration)

tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

input_sequence = "beef, salt, pepper"
input_ids = tokenizer.encode(input_sequence, return_tensors='pt')

# Create the attention mask
attention_mask = torch.ones(input_ids.shape, device=device)

model = model.to(device)

# Combine both sampling techniques
sample_outputs = model.generate(
    input_ids.to(device),
    attention_mask=attention_mask.to(device),
    do_sample=True,
    max_length=120,
    top_k=50,
    top_p=0.85,
    num_return_sequences=3
)

print("Output:\n" + 100 * '-')
for i, sample_output in enumerate(sample_outputs):
    print("{}: {}...".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
    print('  ---')

df_recipes = pd.read_csv('./data/recipes_1000.csv')
df_recipes.reset_index(drop=True, inplace=True)

# df_recipes = df_recipes.iloc[:600]
print(list(df_recipes.columns))
print(f"data shape {df_recipes.shape}")

import nltk
import numpy as np

doc_lengths = []

for rec in df_recipes.itertuples():

    # get rough token count distribution
    tokens = nltk.word_tokenize(rec.ingredients + ' ' + rec.instructions)

    doc_lengths.append(len(tokens))

doc_lengths = np.array(doc_lengths)
# the max token length
print(f"% documents > 180 tokens: {round(len(doc_lengths[doc_lengths > 180]) / len(doc_lengths) * 100, 1)}%")
print(f"Average document length: {int(np.average(doc_lengths))}")
def form_string(ingredient,instruction):
    # s = f"<|startoftext|>Ingredients:\n{ingredient.strip()}\n\nInstructions:\n{instruction.strip()}<|endoftext|>"
    s = f"<|startoftext|>Ingredients: {ingredient.strip()}. " \
        f"Instructions: {instruction.strip()}<|endoftext|>"
    return s

def extract_string(recipe):
    str = recipe.replace('<|startoftext|>', '').replace('<|endoftext|>', '')
    inst_pos = str.find('Instructions: ')
    ingredients = str[len('Ingredients: '): inst_pos-1]
    instructions = str[inst_pos+len('Instructions: '):]
    return ingredients, instructions

data = df_recipes.apply(lambda x:form_string(
    x['ingredients'], x['instructions']), axis=1).to_list()
data[0]

tokenizer = GPT2TokenizerFast.from_pretrained(model_name,
                                              bos_token='<|startoftext|>',
                                              eos_token='<|endoftext|>',
                                              unk_token='<|unknown|>',
                                              pad_token='<|pad|>'
                                             )

vocab_list = sorted(tokenizer.vocab.items(), key=lambda x:x[1])
for i in range(5555, 5566):
    print(vocab_list[i])

print("The max model length is {} for this model".format(tokenizer.model_max_length))
print("The end of sequence token {} has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id), tokenizer.eos_token_id))
print("The beginning of sequence token {} has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.bos_token_id), tokenizer.bos_token_id))
print("The unknown token {} has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.unk_token_id), tokenizer.unk_token_id))
print("The padding token {} has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id), tokenizer.pad_token_id))

# GPT2 is a large model. Increasing the batch size above 2 has lead to out of memory problems.
batch_size = 2
max_length = 180  # maximum sentence length

# standard PyTorch approach of loading data in using a Dataset class.
class RecipeDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.input_ids = []
        self.attn_masks = []
        self.origin_ingredients = []
        self.origin_instructions = []

        for recipe in data:
            encodings = tokenizer.encode_plus(recipe,
                                              truncation=True,
                                              padding='max_length',
                                              max_length=max_length,
                                              return_tensors='pt'       # return PyTorch tensor
                                             )
            self.input_ids.append(torch.squeeze(encodings['input_ids'],0))
            # attention_mask tells model not to incorporate these PAD tokens into its interpretation of the sentence
            self.attn_masks.append(torch.squeeze(encodings['attention_mask'],0))
            ingredients, instructions = extract_string(recipe)
            self.origin_ingredients.append(ingredients)
            self.origin_instructions.append(instructions)


    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.input_ids[idx], self.attn_masks[idx], self.origin_ingredients[idx], self.origin_instructions[idx]
    

dataset = RecipeDataset(data, tokenizer)

# Split into training and validation sets
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

print(f"dataset size {dataset.__len__()}")
print(f"dataset[0]: \n  input_ids: {dataset[0][0]}\n  attn_masks: {dataset[0][1]}")

# Create the DataLoaders for our training and validation datasets.
# We'll take training samples in random order.
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )


configuration = GPT2Config.from_pretrained(model_name, output_hidden_states=False)
model = GPT2LMHeadModel.from_pretrained(model_name, config=configuration)
model = model.to(device)
print(f"Weight shape {model.transformer.wte.weight.shape}")
# this step is necessary because I've added some tokens (bos_token, etc.) to the embeddings
# otherwise the tokenizer and model tensors won't match up
model.resize_token_embeddings(len(tokenizer))
print(f"Number of tokens: {len(tokenizer)}")

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

word_embeddings = model.transformer.wte.weight # Word Token Embeddings

print(word_embeddings.shape)

epochs = 3
learning_rate = 2e-5
warmup_steps = 1e2
# The epsilon parameter eps = 1e-8 is “a very small number to prevent any division by zero in the implementation”
epsilon = 1e-8
# optim = Adam(model.parameters(), lr=5e-5)
optim = AdamW(model.parameters(), lr = learning_rate, eps = epsilon)

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


# Total number of training steps is [number of batches] x [number of epochs].
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
# This changes the learning rate as the training loop progresses
scheduler = get_linear_schedule_with_warmup(optim,
                                            num_warmup_steps = warmup_steps,
                                            num_training_steps = total_steps)

def infer(prompt):
    input = f"<|startoftext|>Ingredients: {prompt.strip()}"
    input = tokenizer(input, return_tensors="pt")
    input_ids      = input["input_ids"]
    attention_mask = input["attention_mask"]

    output = model.generate(input_ids.to(device),
                            attention_mask=attention_mask.to(device),
                            max_new_tokens=max_length,
                            # temperature = 0.5,
                            do_sample = True, top_k = 50, top_p = 0.85)
                            # num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    return output

total_t0 = time.time()

training_stats = []

for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()

    total_train_loss = 0

    model.train()  # `train` just changes the *mode* (train vs. eval), it doesn't *perform* the training.

    for step, batch in enumerate(train_dataloader):     # step from enumerate() = number of batches

        b_input_ids = batch[0].to(device)   # tokens (of multiple documents in a batch)
        b_labels    = batch[0].to(device)
        b_masks     = batch[1].to(device)   # mask of [1] for a real word, [0] for a pad

        model.zero_grad()
        # loss = model(X.to(device), attention_mask=a.to(device), labels=X.to(device)).loss
        outputs = model(  input_ids = b_input_ids,
                          labels = b_labels,
                          attention_mask = b_masks,
                          token_type_ids = None
                        )

        loss = outputs[0]

        batch_loss = loss.item()
        total_train_loss += batch_loss

        # Get sample every x batches.
        if step % 100 == 0 and not step == 0:

            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader), batch_loss, elapsed))

            model.eval()

            sample_output = infer("eggs, flour, butter, sugar")
            print(sample_output)

            # `train` just changes the *mode* (train vs. eval), it doesn't *perform* the training.
            model.train()

        loss.backward()
        optim.step()
        scheduler.step()

    # Calculate the average loss over all the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)

    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))


    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:

        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)

        with torch.no_grad():

            outputs  = model(input_ids = b_input_ids,
                             attention_mask = b_masks,
                             labels = b_labels)

            loss = outputs[0]

        batch_loss = loss.item()
        total_eval_loss += batch_loss

    avg_val_loss = total_eval_loss / len(validation_dataloader)

    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

# Create a DataFrame from our training statistics.
df_stats = pd.DataFrame(data=training_stats)

# Use the 'epoch' as the row index.
df_stats = df_stats.set_index('epoch')
df_stats
print("Saving model to %s" % model_save_path)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
# model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

model = GPT2LMHeadModel.from_pretrained(model_save_path)
tokenizer = GPT2TokenizerFast.from_pretrained(model_save_path)
model.to(device)
