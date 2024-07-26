import pandas as pd

# Load the data
df = pd.read_json('./data/kant_prompts_and_completions.json', lines=True)
df
print(df)