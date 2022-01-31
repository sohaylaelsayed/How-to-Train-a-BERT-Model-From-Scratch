from transformers import pipeline
from dataset import geting_data
from tokens import Tokenizer
from train import training

def result():
    #geting_data()
    #Tokenizer.custom_tokenizer()
    create_train = training()
    return (create_train)


def test():
    result()

    fill = pipeline('fill-mask', model='filiberto', tokenizer='filiberto')

    #Some weights of RobertaModel were not initialized from the model checkpoint at filiberto and are newly initialized: ['roberta.pooler.dense.weight', 'roberta.pooler.dense.bias']
    #You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.

    fill(f'good, {fill.tokenizer.mask_token} ?')

test()
