from datasets import load_dataset
from tqdm.auto import tqdm  #tqdm is a Python library that allows you to output a smart progress bar by wrapping around any iterable.

def geting_data():

    dataset = load_dataset ('onestop_english' , ' default')  #Hugging Face dataset
    text_data = []

    for sample in tqdm(dataset['train']):
        sample = sample['text'].replace('\n', '')
        text_data.append(sample)
        if len(text_data) == 564:
            # once we git the <1K mark, save to file
            with open(f'data/text.txt', 'w', encoding='utf-8') as fp:
                fp.write('\n'.join(text_data))