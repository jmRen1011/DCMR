import os
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments, 
    HfArgumentParser, 
    Seq2SeqTrainingArguments
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    PeftModel, 
    PeftConfig
)
from utils import print_trainable_parameters
from dataset import CustomDataset, DataForm
from finetuning_args import ScriptArguments
from trl import (
    SFTTrainer, 
    SFTConfig, 
    DataCollatorForCompletionOnlyLM
    )
from typing import cast

class ModelTrainer:
    def __init__(self, script_args, training_args):
        self.script_args = script_args
        self.training_args = training_args
        self.response_template = '[/INST]'
        self.instruction_template = '[INST]'
        self.tokenizer = None
        self.model = None
        self.train_data = None
        self.eval_data = None
        self.data_form = DataForm(prompt_file='/home/rjm/code/data/prompt/train_prompt_v0.1.txt')

    def setup_tokenizer(self):
        token_model_path = "/home/rjm/mine/Mistral-7B-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(token_model_path, padding_side="right")
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def setup_datacollator(self):
        self.data_collator = DataCollatorForCompletionOnlyLM(
            tokenizer=self.tokenizer,
            response_template=self.response_template,
            instruction_template=self.instruction_template
        )

    def load_datasets(self):
        train_file = "./data/sharc_raw/json/sharc_train.json"
        train_doc_file = "./RotatE_sharc_4/parsed/train_snippet_parsed_with_id.json"
        train_tokenized_data = data_form.process(data_file=train_file, doc_file=train_doc_file)
        self.train_data = CustomDataset(train_tokenized_data)
        
        
        dev_file = "./data/sharc_raw/json/sharc_dev.json"
        dev_doc_file = "./RotatE_sharc_4/parsed/dev_snippet_parsed_with_id.json"
        dev_tokenized_data = data_form.process(data_file=dev_file, doc_file=dev_doc_file)
        self.dev_data = CustomDataset(dev_tokenized_data)

    def load_model(self):
        model_path = self.script_args.checkpoint_path if self.script_args.load_checkpoint else "/home/rjm/mine/Mistral-7B-instruct"
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        config = LoraConfig(
            r=self.script_args.lora_r,
            lora_alpha=self.script_args.lora_alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", 
                "up_proj", "down_proj", "lm_head"
            ],
            lora_dropout=self.script_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=["edu_embed", "cross_attn", "cross_attention_layernorm"]
        )
        self.model = get_peft_model(self.model, config)

        if self.training_args.do_predict:
            config = PeftConfig.from_pretrained(self.script_args.checkpoint_path)
            self.model = PeftModel.from_pretrained(self.model, self.script_args.checkpoint_path, config=config)

        self.model.to("cuda")
        print_trainable_parameters(self.model)

    def compute_metrics(self, pred):
        preds, labels = pred
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        def extract_label(text):
            if "__YES__" in text:
                return "__YES__"
            elif "__NO__" in text:
                return "__NO__"
            elif "__INQUIRE__" in text:
                return "__INQUIRE__"
            elif "__IRRELEVANT__" in text:
                return "__IRRELEVANT__"
            else:
                return "UNKNOWN"
        
        extracted_preds = [extract_label(pred) for pred in decoded_preds]
        extracted_labels = [extract_label(label) for label in decoded_labels]
        
        label_mapping = {"__YES__": 0, "__NO__": 1, "__INQUIRE__": 2, "__IRRELEVANT__": 3, "UNKNOWN": -1}
        numerical_preds = [label_mapping[label] for label in extracted_preds]
        numerical_labels = [label_mapping[label] for label in extracted_labels]
        
        valid_indices = [i for i, label in enumerate(numerical_labels) if label != -1]
        filtered_preds = [numerical_preds[i] for i in valid_indices]
        filtered_labels = [numerical_labels[i] for i in valid_indices]
        
        accuracy = accuracy_score(filtered_labels, filtered_preds)
        precision, recall, f1_micro, _ = precision_recall_fscore_support(filtered_labels, filtered_preds, average='micro')
        _, _, f1_macro, _ = precision_recall_fscore_support(filtered_labels, filtered_preds, average='macro')

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro
        }

    def train(self):
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.train_data,
            eval_dataset=self.eval_data,
            args=self.training_args,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        self.model.config.use_cache = False
        trainer.accelerator.print(f"{trainer.model}")

        if self.training_args.do_train:
            if self.script_args.load_checkpoint:
                trainer.train(resume_from_checkpoint=True)
            else:
                trainer.train()

        elif self.training_args.do_predict:
            predictions, labels, metrics = trainer.predict(self.eval_data, metric_key_prefix="predict")
            trainer.log_metrics("predict", metrics)
            trainer.save_metrics("predict", metrics)

            predictions = np.argmax(predictions, axis=1)
            output_predict_file = os.path.join(self.training_args.output_dir, "predictions.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        writer.write(f"{index}\t{item}\n")

def main():
    parser = HfArgumentParser([ScriptArguments, Seq2SeqTrainingArguments])
    script_args, training_args = parser.parse_args_into_dataclasses()
    args = cast(ScriptArguments, script_args)
    training_args = cast(Seq2SeqTrainingArguments, training_args)

    trainer = ModelTrainer(args, training_args)
    trainer.setup_tokenizer()
    trainer.setup_datacollator()
    trainer.load_datasets()
    trainer.load_model()
    trainer.train()

if __name__ == '__main__':
    main()
