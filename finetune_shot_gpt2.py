#WRTTIEN IN GOOGLE COLAB, AND USED A100 80GB GPU

# Install required packages (first cell in Colab)
!pip install -q transformers datasets accelerate evaluate sentencepiece
# If you need a specific torch version, install it explicitly (only if necessary)
# !pip install -q torch==1.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html


#IMPORTS AND GPU CHECKSimport os
import math
import random
from pprint import pprint
import torch

from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed
)

from datasets import load_dataset
import evaluate

SEED = 42
set_seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    
#LOADING THE DATASET and THE GPT2 MODEL
raw = load_dataset("ag_news")

print("Dataset splits: ", raw.keys())
print("Example record:")
pprint(raw["train"][0])

"""PREPARING TRAINING/VALIDATION SPLIT AND KEEPING TEXT ONLY"""
train_ds = raw["train"].remove_columns("label")
val_ds = raw["test"].remove_columns("label")

print("Train size:", len(train_ds),
      "Validation size:", len(val_ds))

print("Example Text:", train_ds[0]["text"])


##LOADING THE PRETRAINED GPT2-SMALL MODEL
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

MODEL_NAME = "gpt2"

#"loading the tokenizer"

tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
  tokenizer.pad_token = tokenizer.eos_token
"""Loading the gpt2 model"""
base_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
base_model.resize_token_embeddings(len(tokenizer))
base_model.config.pad_token_id = tokenizer.pad_token_id
base_model.to(device)


##TOKENIZING THE DATASET
def tokenize_batch(examples):
    return tokenizer(examples["text"], truncation=True)

# Appling tokenizer to train and validation sets
tokenized_train = train_ds.map(tokenize_batch, batched=True, remove_columns=train_ds.column_names)
tokenized_val = val_ds.map(tokenize_batch, batched=True, remove_columns=val_ds.column_names)

# Grouping texts into fixed-size chunks (block_size = 512 for A100)
block_size = 512

def group_texts(examples):
    # Flattening list of lists
    all_ids = sum(examples["input_ids"], [])
    total_len = (len(all_ids) // block_size) * block_size
    chunks = [all_ids[i:i+block_size] for i in range(0, total_len, block_size)]
    return {
        "input_ids": chunks,
        "attention_mask": [[1]*block_size for _ in range(len(chunks))]
    }

#maping grouping function
lm_train = tokenized_train.map(group_texts, batched=True, batch_size=1000)
lm_val = tokenized_val.map(group_texts, batched=True, batch_size=1000)

print("Training blocks:", len(lm_train))
print("Validation blocks:", len(lm_val))



##FEW SHOTS PROMPTING (BASELINE GPT2, BEFORE FINE TUNING)
def build_prompt(k, dataset, idx=0):
    chosen = dataset.select(range(k+1))
    prompt = ""
    for i in range(k):
        prompt += f"News: {chosen[i]['text']}\n\n"
    prompt += f"News: {chosen[k]['text']}\n"
    return prompt

def generate_text(model, tokenizer, prompt, max_new_tokens=40):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    gen_ids = model.generate(
        input_ids,
        do_sample=True,
        top_p=0.9,              # fixed
        temperature=0.7,        # fixed
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id
    )
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return gen_text[len(prompt):].strip()

# Testing the pretrained GPT-2 (before fine-tuning) with formatted output
for k in [0, 1, 3]:
    prompt = build_prompt(k, val_ds, idx=5)
    gen_output = generate_text(base_model, tokenizer, prompt)

    print("="*100)
    print(f" FEW-SHOT TEST (k={k})")
    print("="*100)

    print("\nPrompt given to model:\n")
    print(prompt.strip())

    print("\nModel Generated:\n")
    print(gen_output.strip())
    print("\n\n")
    
    
## OPTIONAL -> INTERACTIVE PROMPTING
def test_few_shot(model, tokenizer, prompt_texts, max_new_tokens=50):
    """
    model: GPT model (base or fine-tuned)
    tokenizer: GPT tokenizer
    prompt_texts: list of example strings ["News: ...", "News: ..."]
    """
    # Building the prompt from user-provided examples
    prompt = "\n\n".join([f"News: {t}" for t in prompt_texts])

    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(device)
    attn_mask = inputs["attention_mask"].to(device)

    gen_ids = model.generate(
        input_ids,
        attention_mask=attn_mask,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,
        min_new_tokens = 40
    )
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    print("="*100)
    print("Prompt given:\n")
    print(prompt)
    print("\nModel continuation:\n")
    print(gen_text[len(prompt):].strip())
    print("="*100)

custom_examples = [
    "AI startup raises $50M to build next-gen language model",
    "New vaccine shows promise in clinical trials",
    "Stock markets rally after positive earnings reports",
]

test_few_shot(base_model, tokenizer, custom_examples)


## FINE TUNING GPT2 
print("FINE TUNING THE GPT2 ON THE AG-NEWS DATASET\n\n")
!rm -rf ./gpt2-finetuned-agnews  # clear old checkpoints

from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments, GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
model.to(device)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./gpt2-finetuned-agnews",
    overwrite_output_dir=True,
    num_train_epochs=20,
    per_device_train_batch_size=64,
    gradient_accumulation_steps=4,
    save_strategy="epoch",
    logging_steps=100,
    learning_rate=3e-5, 
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    report_to="none",
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_train,
    eval_dataset=lm_val,
    data_collator=data_collator,
)
trainer.train() 
""" In my case, this is the output I got is

TrainOutput(
	global_step=960, 
	training_loss=3.2326313972473146, 
	metrics={
		'train_runtime': 1858.0543, 
		'train_samples_per_second': 130.76, 
		'train_steps_per_second': 0.517, 
		'total_flos': 6.348351209472e+16, 
		'train_loss': 3.2326313972473146, 
		'epoch': 20.0}
	)
Training time = 32.06 minutes

"""

## PERPLEXITY EVALATION (HOW SUPRRISED THE MODOEL IS AFTER SEEING NEW WORDS)
import math
eval_results = trainer.evaluate()
eval_loss = eval_results["eval_loss"]
perplexity = math.exp(eval_loss)
print(f"Validation loss: {eval_loss:.4f} → Perplexity: {perplexity:.2f}")

"""
O/P -> Validation loss: 3.0742 → Perplexity: 21.63 
"""

"""
TESTTING PRE TRAINED GPT2 WITH SHOTS AND FINETUNED GPT2 ON AG NEWS DATASET
"""
test_idx = 10  #samples
for k in [0, 1, 3]:
    prompt = build_prompt(k, val_ds, idx=test_idx)
    base_out = generate_text(base_model, tokenizer, prompt)
    ft_out = generate_text(ft_model, tokenizer, prompt)

    print("="*100)
    print(f"Prompt (K={k}):\n{prompt[:300]}...\n")
    print("Pretrained GPT-2 →", base_out, "\n")
    print("Fine-tuned GPT-2 →", ft_out)
    print("="*100, "\n")

"""
OUTPUT:

====================================================================================================
Prompt (K=0):
News: Fears for T N pension after talks Unions representing workers at Turner   Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul.
...

Pretrained GPT-2 → TNS said it had reached an agreement with the parent company but that it would not discuss any details until a final deal was reached.

The company has also taken its concerns about the future 

Fine-tuned GPT-2 → Tennis: Toni wins, Pirlo beats Serena in second round Toni Pirlo won her fourth straight WTA Tour title and Serena Williams beat her in the second round of
==================================================================================================== 
====================================================================================================
Prompt (K=1):
News: Fears for T N pension after talks Unions representing workers at Turner   Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul.

News: The Race is On: Second Private Team Sets Launch Date for Human Spaceflight (SPACE.com) SPACE.com - TORONTO, Canada -- A secon...

Pretrained GPT-2 → The team, led by the Canadian Space Agency (CSA), will enter into a three-year contract with the company to launch the first-ever manned spaceflight from Cape Canaveral Air Force Station 

Fine-tuned GPT-2 → News: NASA Releases First Shuttle Flight of Columbia Crew (SPACE.com) SPACE.com - NASA #39;s space shuttle Columbia is set to launch Tuesday, Oct. 16, 2005
==================================================================================================== 

====================================================================================================
Prompt (K=3):
News: Fears for T N pension after talks Unions representing workers at Turner   Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul.

News: The Race is On: Second Private Team Sets Launch Date for Human Spaceflight (SPACE.com) SPACE.com - TORONTO, Canada -- A secon...

Pretrained GPT-2 → News: 'Trying to Survive' is a 'real struggle' (AP) AP - The Canadian Space Agency is trying to cope with the rapid rise of wildfires in the Western United States. 

Fine-tuned GPT-2 → News: NASA to Launch Shuttle Into Space Station (AP) AP - The United States has successfully completed the first of three manned flights into space, NASA announced Tuesday, and it will launch a second
==================================================================================================== 

"""

#BLEU SCORE EVALUAION
import evaluate

bleu = evaluate.load("bleu")
def evaluate_bleu(model, tokenizer, dataset, k=0, num_samples=100):
    preds, refs = [], []
    for i in range(num_samples):
        prompt = build_prompt(k, dataset, idx=i)
        gen = generate_text(model, tokenizer, prompt)
        preds.append(gen)
        refs.append([dataset[i]["text"]])  # reference is the real news text
    return bleu.compute(predictions=preds, references=refs)

print("\nBaseline GPT-2 BLEU:", evaluate_bleu(base_model, tokenizer, val_ds, k=0, num_samples=100))
print("Fine-tuned GPT-2 BLEU:", evaluate_bleu(ft_model, tokenizer, val_ds, k=0, num_samples=100))

"""
OUTPUT: 
Baseline GPT-2 BLEU: {
	'bleu': 0.0, 
	'precisions': [0.16562778272484416, 0.0061180789232181095, 0.0003155569580309246, 0.0], 
	'brevity_penalty': 0.6012428752612912, 
	'length_ratio': 0.6627975604957702, 
	'translation_length': 3369, 
	'reference_length': 5083}



Fine-tuned GPT-2 BLEU: {
	'bleu': 0.008069041768349354, 	
	'precisions': [0.15918135304150086, 0.011702750146284377, 0.004822182037371911, 0.0027967681789931634], 
	'brevity_penalty': 0.6409171822492856, 
	'length_ratio': 0.6921109580956128, 
	'translation_length': 3518, 
	'reference_length': 5083}

"""








































































