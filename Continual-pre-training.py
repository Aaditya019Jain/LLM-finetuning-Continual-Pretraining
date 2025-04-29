import os
import logging
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    default_data_collator,
    get_scheduler,
    AdamW,
)
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate import FullyShardedDataParallelPlugin
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from functools import partial
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

def setup_logger(log_file: str):
    """Setup logger to log both to console and a log file."""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def main():
    # Setup logger
    log_file = "training_log.log"
    logger = setup_logger(log_file)
    logger.info("Starting the fine-tuning script for causal language modeling on CulturaX.")

    auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=1e6)

    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
        auto_wrap_policy=auto_wrap_policy
    )

    torch.distributed.init_process_group(backend="nccl")

    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
    set_seed(1234)

    model_name = "meta-llama/Llama-3.1-8B"
    logger.info(f"Loading tokenizer for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("Configuring LoRA.")
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj']
    )

    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True
    )
    model.config.pretraining_tp = 1
    model = get_peft_model(model, peft_config)

    # Prepare model with Accelerator
    model = accelerator.prepare(model)

    logger.info("Loading dataset: CulturaX")
    dataset = load_dataset("uonlp/CulturaX", "bn", split="train")

    def preprocess_function(examples):
        """Tokenize text and prepare for training."""
        outputs = tokenizer(
            examples['text'],
            truncation=True,
            max_length=2048,
            padding=True,
            return_tensors=None  # Return lists instead of tensors
        )
        
        # Add labels for causal language modeling
        outputs["labels"] = outputs["input_ids"].copy()
        
        return outputs

    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )

    # Set format to PyTorch
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    train_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=2,
        collate_fn=default_data_collator,
        shuffle=True
    )
    train_dataloader = accelerator.prepare(train_dataloader)

    optimizer = AdamW(model.parameters(), lr=2e-4)
    optimizer = accelerator.prepare(optimizer)

    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(train_dataloader) * 1,
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)

    logger.info("Starting training.")
    model.train()
    
    for epoch in range(1):
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            if step % 10 == 0:
                avg_loss = total_loss / (step + 1)
                logger.info(f"Epoch {epoch}, Step {step}, Average Loss: {avg_loss:.4f}")

    logger.info("Saving the final model.")
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained("./train_output", save_function=accelerator.save)
    logger.info("Training script completed successfully.")

if __name__ == "__main__":
    main()
