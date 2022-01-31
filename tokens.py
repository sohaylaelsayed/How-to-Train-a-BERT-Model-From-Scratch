from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaTokenizer
import os

class Tokenizer():

    def custom_tokenizer():

        path = ('data/text.txt') 

        #Now we move onto training the tokenizer. We use a byte-level Byte-pair encoding (BPE) tokenizer. 
        #This allows us to build the vocabulary from an alphabet of single bytes, meaning all words will be decomposable into tokens.


        tokenizer = ByteLevelBPETokenizer()

        tokenizer.train(files=path, vocab_size=10_0, min_frequency=2,
                        special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])

        #Our tokenizer is now ready, and we can save it file for later use:

        os.mkdir('./filiberto')

        tokenizer.save_model('filiberto')


    def initial_tokenizer():

        # initialize the tokenizer using the tokenizer we initialized and saved to file
        tokenizer = RobertaTokenizer.from_pretrained('filiberto', max_len=512)
        return(tokenizer)
