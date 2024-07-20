import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

display_architecture=False
model = T5ForConditionalGeneration.from_pretrained('t5-large')
tokenizer = T5Tokenizer.from_pretrained('t5-large')
device = torch.device('cpu')

if display_architecture==True:
 print(model.config)

if display_architecture==True:
    print(model.encoder)

def summarize(text, maximum_length):
  preprocess_text = text.strip().replace("\n","")
  t5_prepared_Text = "summarize: "+preprocess_text
  print ("Preprocessed and prepared text: \n", t5_prepared_Text)

  tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)

  # summmarize 
  summary_ids = model.generate(tokenized_text,
                                      num_beams=4,
                                      no_repeat_ngram_size=2,
                                      min_length=30,
                                      max_length=maximum_length,
                                      early_stopping=True)

  output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
  return output



text="""
The United States Declaration of Independence was the first Etext released by Project Gutenberg, early in 1971.  The title was stored in an emailed instruction set which required a tape or diskpack be hand mounted for retrieval.  The diskpack was the size of a large cake in a cake carrier, cost $1500, and contained 5 megabytes, of which this file took 1-2%.  Two tape backups were kept plus one on paper tape.  The 10,000 files we hope to have online by the end of 2001 should take about 1-2% of a comparably priced drive in 2001. """

print("Number of characters:",len(text))
summary=summarize(text,54)
print ("\n\nSummarized text: \n",summary)



text="""
Microsoft says it estimates that 8.5m computers around the world were disabled by the global IT outage.
It’s the first time that a number has been put on the incident, which is still causing problems around the world.
The glitch came from a cyber security company called CrowdStrike which sent out a corrupted software update to its huge number of customers.
Microsoft, which is helping customers recover said in a blog post: "we currently estimate that CrowdStrike’s update affected 8.5 million Windows devices."
"""

print("Number of characters:",len(text))
summary=summarize(text,54)
print ("\n\nSummarized text: \n",summary)