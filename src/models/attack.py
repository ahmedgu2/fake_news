import pandas as pd
import hydra
import torch
from sklearn.model_selection import train_test_split
import transformers
import textattack
from omegaconf import DictConfig, OmegaConf


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

    X_train, X_val, y_train, y_val = prepare_data(dataset_name=cfg['dataset']['name'], dataset_path=cfg['dataset']['path'])
    model = transformers.AutoModelForSequenceClassification.from_pretrained(cfg['model']['trained_model'])
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg['model']['model_name'],
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
    #prepare dataset
    attack_dataset = [(input[:512], int(output)) for input, output in zip(X_val.values, y_val.values) if len(input) > 10]
    dataset = textattack.datasets.Dataset(attack_dataset)
    attack = textattack.attack_recipes.bae_garg_2019.BAEGarg2019.build(model_wrapper)
    # Attack 20 samples with CSV logging and checkpoint saved every 5 interval
    attack_args = textattack.AttackArgs(
        num_examples=200, 
        log_to_csv=f"attack_logs/BAEGarg2019_{cfg['dataset']['name']}_log.csv", 
        checkpoint_interval=5, 
        checkpoint_dir=f"checkpoints/BAEGarg2019_{cfg['dataset']['name']}/", 
        disable_stdout=True,
        query_budget=1000,
        shuffle=True, 
        random_seed=42
    )
    attacker = textattack.Attacker(attack, dataset, attack_args)
    attacker.attack_dataset()


if __name__ == '__main__':
    main()