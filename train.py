from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
import torch 
from transformers import AdamW
from tqdm.auto import tqdm
from piplines import InitializeDataset




def initialize_model():

    config = RobertaConfig(
        vocab_size=10_0,  # we align this to the tokenizer vocab_size
        max_position_embeddings= 100 ,
        hidden_size=100,
        num_attention_heads=5,
        num_hidden_layers=3,
        type_vocab_size=1
    )


    model = RobertaForMaskedLM(config)
    return(model)

def train_prep():
    device = torch.device('cuda') 
    # and move our model over to the selected device
    model = initialize_model()
    model.to(device)

    #Activate the training mode of our model, and initialize our optimizer (Adam with weighted decay - reduces chance of overfitting).
    # activate training mode
    model.train()
    # initialize optimizer
    optim = AdamW(model.parameters(), lr=1e-4)
    return(optim,device,model)


def training():

    epochs = 2
    optim,device,model = train_prep()
    loader = InitializeDataset.intial_data()
    for epoch in range(epochs):
        # setup loop with TQDM and dataloader
        loop = tqdm(loader, leave=True)
        for batch in loop:
            # initialize calculated gradients (from prev step)
            optim.zero_grad()
            # pull all tensor batches required for training
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # process
            outputs = model(input_ids, attention_mask=attention_mask,labels=labels)
            # extract loss
            loss = outputs.loss
            # calculate loss for every parameter that needs grad update
            loss.backward()
            # update parameters
            optim.step()
            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}')

    model.save_pretrained('./filiberto')  # and don't forget to save filiBE