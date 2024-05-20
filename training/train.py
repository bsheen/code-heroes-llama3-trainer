import os
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TrainingArguments
from trl import SFTTrainer

max_seq_length = 2048 # Supports RoPE Scaling interally, so choose any!

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",
    "unsloth/gemma-7b-it-bnb-4bit", # Instruct version of Gemma 7b
    "unsloth/gemma-2b-bnb-4bit",
    "unsloth/gemma-2b-it-bnb-4bit", # Instruct version of Gemma 2b
    "unsloth/llama-3-8b-bnb-4bit", # [NEW] 15 Trillion token Llama-3
    "unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = os.environ["HF_INPUT_MODEL"],
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

eos_token = "<|eot_id|>"
llama3_template = \
    "{{ bos_token }}"\
    "{% for message in messages %}"\
        "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}"\
    "{% endfor %}"\
    "{{ eos_token }}"

tokenizer = get_chat_template(
    tokenizer,
    chat_template = (llama3_template, eos_token,)
)

special_tokens_dict = {"eos_token": "<|eot_id|>"}
tokenizer.add_special_tokens(special_tokens_dict)

def formatting_prompts_func(examples):
    chats = examples["chats"]
    texts = [tokenizer.apply_chat_template(chat, tokenize = False, add_generation_prompt = False) for chat in chats]
    return { "text" : texts, }

from datasets import load_dataset
dataset_id = os.environ["HF_ACCOUNT"] + "/" + os.environ["HF_DATASET"]
dataset_train = load_dataset(dataset_id, split = "train")
dataset_train = dataset_train.map(formatting_prompts_func, batched = True,)
dataset_test = load_dataset(dataset_id, split = "test")
dataset_test = dataset_test.map(formatting_prompts_func, batched = True,)

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    max_seq_length = max_seq_length,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    dataset_text_field = "text",
    train_dataset = dataset_train,
    eval_dataset = dataset_test,
    max_seq_length = max_seq_length,
    args = TrainingArguments(
        num_train_epochs=1,
        per_device_train_batch_size = 16,
        gradient_accumulation_steps = 32,
        warmup_steps = 10,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = "outputs",
        optim = "adamw_8bit",
        seed = 3407,
        save_strategy = "steps",
        save_steps = 40,
        evaluation_strategy = "steps",
        eval_steps = 20
    ),
)
trainer.train()

model.save_pretrained_merged(os.environ["HF_OUTPUT_MODEL"] + "_merged_16bit", tokenizer, save_method = "merged_16bit",)
model.push_to_hub_merged(os.environ["HF_ACCOUNT"] + "/" + os.environ["HF_OUTPUT_MODEL"], tokenizer, save_method = "merged_16bit", token = os.environ["HF_TOKEN"])

model.save_pretrained_gguf(os.environ["HF_OUTPUT_MODEL"] + "_q4_k_m", tokenizer, quantization_method = "q4_k_m")
model.push_to_hub_gguf(os.environ["HF_ACCOUNT"] + "/" + os.environ["HF_OUTPUT_MODEL"], tokenizer, quantization_method = "q4_k_m", token = os.environ["HF_TOKEN"])