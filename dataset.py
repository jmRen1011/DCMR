import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DataCollatorForCompletionOnlyLM
from torch.utils.data import DataLoader, Dataset
import torch

class CustomDataset(Dataset):
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        return {
            'input_ids': self.tokenized_data[idx]['input_ids'].squeeze(0),
            'attention_mask': self.tokenized_data[idx]['attention_mask'].squeeze(0),
            'edu_ids': self.tokenized_data[idx]['edu_ids'].squeeze(0),
        }
    
class DataForm():
    def __init__(self, prompt_file):
        with open(prompt_file, 'r', encoding='utf-8') as fp:
            self.prompt = "\n".join(fp.readlines())
        self.tokenizer = AutoTokenizer.from_pretrained("/home/rjm/mine/Mistral-7B-instruct", padding_side="right")
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        
        # response_template = '[/INST]'
        # instruction_template = '[INST]'
        # self.data_collator = DataCollatorForCompletionOnlyLM(
        #     tokenizer=self.tokenizer,
        #     response_template=response_template,
        #     instruction_template=instruction_template
        # )
    
    def process_answer(self, answer):
        if answer == "Yes":
            answer = "__Yes__Yes, it's ok."
        elif answer == "No":
            answer = "__No__No, it's not allowed."
        elif answer == "Irrelevant":
            answer = "__Irrelevant__This question is not relevant to the provided information."
        else:
            answer = "__Inquire__" + answer
        return answer

    def form_data(self, data):
        
        question = data["question"]
        snippet = data["snippet"]
        scenario = data["scenario"]
        if not scenario:
            scenario = "Empty"

        input = []
        content = self.prompt.replace("#Question#", question).replace("#Snippet#", snippet).replace("#Scenario#", scenario)
        input.append({"role": "user", "content": content})
        
        history = data["history"]
        if not history:
            answer = self.process_answer(data["answer"])
            input.append({"role": "assistant", "content": answer})
            return input
        else:
            for key, item in history.items():      
                input.append({"role":"assistant", "role": self.process_answer(item["follow_up_question"])})
                input.append({"role":"assistant", "role": self.process_answer(item["follow_up_answer"])})
            answer = self.process_answer(data["answer"])
            input.append({"role": "assistant", "content": answer})
            return input
            
    def process(self, data_file, doc_file):
        with open(data_file, 'r', encoding='utf-8') as fp:
            alldata = json.load(fp)
        with open(doc_file, 'r', encoding='utf-8') as fp:
            alldoc = json.load(fp)

        tokenized_data = []
        for data, doc in zip(alldata, alldoc):
            conversation = self.form_data(data)
            applied = self.tokenizer.apply_chat_template(conversation, tokenize=False)
            tokenized = self.tokenizer(applied, padding='max_length', truncation=True, return_tensors='pt')
            edu_id = doc[data["tree_id"]]["enity_ids"]
            tokenized["edu_ids"] = torch.LongTensor([edu_id])
            tokenized_data.append(tokenized)
        
        return tokenized_data

        
        


        
