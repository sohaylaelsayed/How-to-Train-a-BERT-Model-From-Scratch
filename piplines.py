
import torch
from tokens import Tokenizer 

class Piplines():


    def mlm(tensor):
        # create random array of floats with equal dims to input_ids
        rand = torch.rand(tensor.shape)
        # mask random 15% where token is not 0 [PAD], 1 [CLS], or 2 [SEP]
        mask_arr = (rand < .15) * (tensor >2) 
        # loop through each row in input_ids tensor (cannot do in parallel)
        for i in range(tensor.shape[0]):
            # get indices of mask positions from mask array
            selection = torch.flatten(mask_arr[i].nonzero()).tolist()
            #We can see the special tokens here, 1 is our [CLS] token, 2 our [SEP] token, 3 our [MASK] token, and at the end we have two 0 - or [PAD] - tokens.
            tensor[i, selection] = 3  # our custom [MASK] token == 3
        return tensor

   
    def process():
        input_ids = []
        labels = []
        mask = []
        #First, we need to open our file — the same files that we saved as .txt files earlier. 
        # We split each based on newline characters \n as this indicates the individual samples.

        with open('data/text.txt', 'r', encoding='utf-8') as fp:
            lines = fp.read().split('\n')

        tokenizer = Tokenizer.initial_tokenizer()
        batch = tokenizer(lines, max_length=512, padding='max_length', truncation=True,return_tensors = 'pt')
        labels.append(batch.input_ids)
        mask.append(batch.attention_mask)
        input_ids.append(Piplines.mlm(batch.input_ids.detach().clone()))
        mask = torch.cat(mask)
        labels = torch.cat(labels)
        # make copy of labels tensor, this will be input_ids
        input_ids = torch.cat(input_ids)

        encodings = {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels}
        return(encodings)

class Dataset(torch.utils.data.Dataset):

    def __init__(self, encodings):
        # store encodings internally
        self.encodings = encodings

    def __len__(self):
        # return the number of samples
        return self.encodings['input_ids'].shape[0]

    def __getitem__(self, i):
        # return dictionary of input_ids, attention_mask, and labels for index i
        return {key: tensor[i] for key, tensor in self.encodings.items()}
        


class InitializeDataset():

    def intial_data():
        #Next we initialize our Dataset.
        encodings = Piplines.process()
        dataset = Dataset(encodings)

        #And initialize the dataloader, which will load the data into the model during training.

        loader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True)
        return(loader)

