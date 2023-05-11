from st_on_hover_tabs import on_hover_tabs
import streamlit as st
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
st.set_page_config(layout="wide")
import pandas as pd
import transformers
import textattack
from annotated_text import annotated_text
from st_aggrid import AgGrid

# st.header("Custom tab component for on-hover navigation bar")
st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)


@st.cache_resource
def create_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(
        'distilbert-base-uncased',
        padding='max_length',
        truncation=True,
        return_tensors='pt',
        model_max_length=512,
        max_length=512,
    )
    classifier = pipeline(
        'text-classification',
        model='mlruns/0/0d522e876c104bf1a08ad1d908c4e612/artifacts/checkpoint-4800/artifacts/checkpoint-4800',
        tokenizer=tokenizer
    )
    return classifier


def create_attack_dataset(input, output):
    attack_dataset = [(input[:512], int(output))]
    dataset = textattack.datasets.Dataset(attack_dataset)
    return dataset


@st.cache_resource
def load_attack_resources(attack_str='TextFooler'):
    model = transformers.AutoModelForSequenceClassification.from_pretrained('mlruns/0/0d522e876c104bf1a08ad1d908c4e612/artifacts/checkpoint-4800/artifacts/checkpoint-4800')
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        'distilbert-base-uncased',
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
    #prepare dataset
    # input = """
    #     The Tribeca Film Festival Announces Winners - The New York Times,Mekado Murphy,"The Tribeca Film Festival announced its winners Thursday, with movies from   feature directors and seasoned veterans taking prizes. The Founders Award for best narrative feature went to the comedy “Dean,” the directorial debut of Demetri Martin, who also wrote and stars in the movie. The best international feature prize went to “Junction 48,” an Israeli film with a Palestinian rapper at its center. Craig Atkinson’s “Do Not Resist,” about the militarization of police forces in the United States, won the prize for best documentary feature. All three awards come with $20, 000 prizes. Priscilla Anany won the prize for best new narrative director for her Ghanaian film “Children of the Mountain,” about a woman who faces scrutiny when her child is born with birth defects. The Albert Maysles New Documentary Director Prize went to David Feige for “Untouchable,” which looks at the effect of strict sex offender laws in the United States. The Nora Ephron Prize, awarded to a female director or screenwriter, was given to Rachel Tunnard for her dark comedy “Adult Life Skills,” about a young woman who moves back to her hometown and into a shed in her mother’s backyard. The festival continues through April 24, with additional screenings of some of the prizewinning films to be held this weekend. The festival’s audience awards will be announced April 23.
    #     """
    # output = 0
    # train = pd.read_csv('../data/raw/kaggle_fake_news/train.csv', nrows=10)
    # X_val, y_val = train['text'], train['label']
    # attack_dataset = [(input[:512], int(output)) for input, output in zip(X_val.values, y_val.values) if len(input) > 10]
    #attack_dataset = [(input[:512], int(output)), (input[:512], int(output)), (input[:512], int(output)), (input[:512], int(output))]
    # dataset = textattack.datasets.Dataset(attack_dataset)
    if attack_str == 'TextFooler':
        attack = textattack.attack_recipes.textfooler_jin_2019.TextFoolerJin2019.build(model_wrapper)
    else:
        attack = textattack.attack_recipes.bae_garg_2019.BAEGarg2019.build(model_wrapper)
    # Attack 20 samples with CSV logging and checkpoint saved every 5 interval
    attack_args = textattack.AttackArgs(
        num_examples=1, 
        log_to_csv=f"attack_logs/front_attack.csv", 
        checkpoint_interval=5, 
        # checkpoint_dir=f"checkpoints/BAEGarg2019_distilbert-base-uncased/", 
        disable_stdout=True,
        query_budget=3000,
        shuffle=True, 
        random_seed=42
    )
    return attack, attack_args


def create_attack(attack, dataset, attack_args):
    attacker = textattack.Attacker(attack, dataset, attack_args)
    return attacker.attack_dataset()


def create_sidebar():
    with st.sidebar:
        tabs = on_hover_tabs(tabName=['Fake News Detection', 'Adversarial Examples' ], 
                            iconName=['dashboard', 'money'],
                            styles = {'navtab': {'background-color':'#111',
                                                'color': '#818181',
                                                'font-size': '14px',
                                                'transition': '.3s',
                                                'white-space': 'nowrap',
                                                'text-transform': 'uppercase'},
                                    'tabOptionsStyle': {':hover :hover': {'color': 'red',
                                                                    'cursor': 'pointer'}},
                                    'iconStyle':{'position':'fixed',
                                                    'left':'7.5px',
                                                    'text-align': 'left'},
                                    'tabStyle' : {'list-style-type': 'none',
                                                    'margin-bottom': '30px',
                                                    'padding-left': '30px'}},
                            key="1")
    return tabs

if __name__ == '__main__':
     
    tabs = create_sidebar()
    classifier = create_pipeline()

    if tabs =='Fake News Detection':
        st.title('Fake News Classification')
        df = pd.read_csv('data/raw/kaggle_fake_news/train.csv')
        response = AgGrid(
            df.head(10),
            editable=True,
            data_return_mode="filtered_and_sorted",
        )
        input_text = st.text_area("Text to classify")
        pred_bt = st.button('Predict')
        if pred_bt:
            preds = classifier(input_text, padding=True, truncation=True)
            labels = {'LABEL_1' : 'Fake', 'LABEL_0': 'Real'}
            st.text(f"{labels[preds[0]['label']]}")

    elif tabs == 'Adversarial Examples':
        st.title("Adversarial Example Generation")
        df = pd.read_csv('data/raw/kaggle_fake_news/train.csv')
        response = AgGrid(
            df.head(10),
            editable=True,
            data_return_mode="filtered_and_sorted",
        )
        print(response)
        input_text = st.text_area("Example")
        output = st.selectbox("Original Output", ('Fake', 'Real'))
        output = 1 if output == 'Fake' else 0
        method = st.selectbox("Adversarial Attack Method", ('TextFooler', 'Bert-Attack'))
        attack_bt = st.button('Generate adversarial example')
        attack, attack_args = load_attack_resources(method)
        if attack_bt:
            dataset = create_attack_dataset(input_text, output)
            res = create_attack(attack, dataset, attack_args)
            if 'FAILED' in res[0].goal_function_result_str():
                st.markdown("Adversarial example generation :red[Failed]")
            else:
                st.markdown("Adversarial example generation :green[Succeded]")

                adv = pd.read_csv('attack_logs/front_attack.csv')
                orig_split_text = adv['original_text'][0].split(' ')
                adv_split_text = adv['perturbed_text'][0].split(' ')
                changed_word_indices = set()
                for i, w in enumerate(orig_split_text):
                    if '[[' in w and ']]' in w:
                        changed_word_indices.add(i)
                out = [(adv_split_text[i].replace('[[', "").replace(']]', ""), w.replace('[[', "").replace(']]', ""), "#faa") if i in changed_word_indices else " " + w + " "  for i, w in enumerate(orig_split_text)]
                out.append(' ...')
                annotated_text(out)
                st.text_area("Copy this", ' '.join([w.replace('[[', "").replace(']]', "") for w in adv_split_text]))

    elif tabs == 'Economy':
        st.title("Tom")
        st.write('Name of option is {}'.format(tabs))