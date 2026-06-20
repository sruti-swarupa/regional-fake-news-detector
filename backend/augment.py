import pandas as pd
import random

# 1. Load your existing balanced dataset
# (Make sure 'train.csv' is in your backend folder and change column names if different)
try:
    df = pd.read_csv("train.csv")
    print(f"Original dataset size: {len(df)} rows")
except Exception as e:
    print("Make sure train.csv is in this folder! Error:", e)
    exit()

# Separate your text and labels (Assuming columns are named 'text' and 'label')
# Adjust 'text' and 'label' to match your exact column names in train.csv
text_col = "Text" 
label_col = "Label"

augmented_records = []

# Common Odia closing tokens/punctuation to vary text patterns dynamically
odia_endings = [" ।", "!", "।", " ବୋଲି ଜଣାପଡିଛି ।", " ଏହି ସୂଚନା ମିଳିଛି ।"]

print("Augmenting data... Please wait...")
for idx, row in df.iterrows():
    orig_text = str(row[text_col]).strip()
    label = row[label_col]
    
    # Keep the original row
    augmented_records.append({text_col: orig_text, label_col: label})
    
    # Only augment Fake rows (or text that might be under-represented in vocabulary variations)
    if label == 1: # Assuming 1 is Fake
        # Mutation 1: Strip basic punctuation trailing spaces
        text_m1 = orig_text.replace("।", "").strip()
        augmented_records.append({text_col: text_m1, label_col: label})
        
        # Mutation 2: Swap endings to simulate different writing structures
        for ending in random.sample(odia_endings, 2):
            text_m2 = text_m1 + ending
            augmented_records.append({text_col: text_m2, label_col: label})

# 2. Create the expanded dataset
new_df = pd.DataFrame(augmented_records)
# Shuffle the rows so the model learns cleanly
new_df = new_df.sample(frac=1).reset_index(drop=True)

# 3. Save it back
new_df.to_csv("train_expanded.csv", index=False)
print(f"🎉 Success! Your dataset has grown to {len(new_df)} rows without collecting anything new!")