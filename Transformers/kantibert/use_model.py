# Language Modeling with the FillMaskPipeline
from transformers import pipeline
MODEL_DIRECTORY="./content/KantaiBERT"
fill_mask = pipeline(
    "fill-mask",
    model=MODEL_DIRECTORY,
    tokenizer=MODEL_DIRECTORY
)

output = fill_mask("Human thinking involves<mask>.")

print(output)