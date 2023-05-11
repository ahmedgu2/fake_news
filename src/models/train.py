import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import os
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
import evaluate
import numpy as np
os.environ['HF_MLFLOW_LOG_ARTIFACTS'] = "1" # save models as artifact for the expirment


class FakeNewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
    
def compute_metrics(eval_preds):
    metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels) 


def prepare_data(dataset_name, dataset_path):
    train = pd.read_csv(f"{dataset_path}/train.csv")
    #test = pd.read_csv(f"{dataset_name}/test.csv")
    train = train.dropna()
    if dataset_name == 'twitter':
        #encode output
        label_mapping = {'fake': 1, 'real': 0}
        train.label = train.label.map(label_mapping)
        #test.label = test.label.map(label_mapping)
        X_train, X_val, y_train, y_val = train_test_split(train['tweetText'], train['label'], test_size=0.2, random_state=42)
    elif dataset_name == 'kaggle_fake_news':
        X_train, X_val, y_train, y_val = train_test_split(train['text'], train['label'], test_size=0.2, random_state=42)
    else:
        print("No valid dataset name!")
        return None
    return X_train, X_val, y_train, y_val, 


@hydra.main(version_base=None, config_path='../config/', config_name='training')
def main(cfg: DictConfig):
    print('###### Config #######')
    print(OmegaConf.to_yaml(cfg))
    ################# Preprocessing data ######################

    X_train, X_val, y_train, y_val = prepare_data(dataset_name=cfg['dataset']['name'], dataset_path=cfg['dataset']['path'])
    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['model_name'])
    #tokenizer.pad_token = tokenizer.eos_token
    X_train_encoded = tokenizer(
        list(X_train.values),
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=512,
    )
    X_val_encoded = tokenizer(
        list(X_val.values),
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=512,
    )
    train_dataset = FakeNewsDataset(X_train_encoded, y_train.values)
    val_dataset = FakeNewsDataset(X_val_encoded, y_val.values)
    #train_loader = DataLoader(train_dataset, batch_size=32)
    val_loader = DataLoader(val_dataset, batch_size=32)

    ####################### Training #############################
    training_args = TrainingArguments(
        output_dir=f"{cfg['model']['model_artifacts']}/{cfg['model']['model_name']}/{cfg['dataset']['name']}/",          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=8,  # batch size per device during training
        per_device_eval_batch_size=8,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=400,
        evaluation_strategy='steps',
        eval_steps=400,
        load_best_model_at_end=True,
        save_total_limit=3,
        save_steps=400

    )

    model = AutoModelForSequenceClassification.from_pretrained(f"{cfg['model']['model_name']}", num_labels=2)

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,             # evaluation dataset
        compute_metrics=compute_metrics
    )

    trainer.train()


if __name__ == '__main__':
    main()