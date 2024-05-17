
import torch
import torch.nn as nn
from torch import optim
import wandb
from evaluvate import evaluate
from encoder import EncoderRNN
from decoder import DecoderRNN, AttnDecoderRNN
from load_data import get_dataloader,prepareData
import argparse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_PATH = './aksharantar_sampled/aksharantar_sampled/tam/tam_train.csv'
TEST_PATH = "./aksharantar_sampled/aksharantar_sampled/tam/tam_test.csv"
VALID_PATH = "./aksharantar_sampled/aksharantar_sampled/tam/tam_valid.csv"
INPUT_LANG_NAME='ENG'
OUTPUT_LANG_NAME='TAM'


def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):

    total_loss = 0

    encoder.train()
    decoder.train()

    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def train_model(train_dataloader, val_dataloader, encoder, decoder,output_lang ,n_epochs, learning_rate):

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
  
    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        train_loss, train_accuracy = evaluate(train_dataloader, encoder, decoder, criterion,output_lang)
        val_loss, val_accuracy = evaluate(val_dataloader, encoder, decoder, criterion,output_lang)

        wandb.log({"epoch": epoch,
                   "train_loss": train_loss,
                   "train_accuracy": train_accuracy,
                   "val_loss": val_loss,
                   "val_accuracy": val_accuracy})
        print("Epoch: {}, Train Loss: {:.4f}, Train Acc: {:.2f}%, Val Loss: {:.4f}, Val Acc: {:.2f}%".format(
            epoch, train_loss, train_accuracy, val_loss, val_accuracy))
        
def run_model():
    
    parser = argparse.ArgumentParser(description="Command line arguments for RNN for Word Transliteration")
    
    parser.add_argument("-wp","--wandb_project",type=str,default="dl_assignment3", help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument("-we","--wandb_entity",type=str,default="cs23m052", help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
    parser.add_argument("-e","--epochs",type=int,default=15, help="Number of epochs to train neural network.")
    parser.add_argument("-b","--batch_size",default=128,type=int, help="Batch size used to train neural network.")
    parser.add_argument("-lr","--learning_rate",default=0.001,type=float, help="Learning rate used to optimize model parameters")
    parser.add_argument("-dpo","--dropout",default=0.2,type=float, help="Dropout for the RNN model")
    parser.add_argument("-nhl","--num_layers",default=1,type=int, help="Number of hidden layers used in Encoder and Decoder of RNN.")
    parser.add_argument("-sz","--hidden_size",default=128, type=int,help="Number of hidden neurons in a RNN.")
    parser.add_argument("-trp","--train_path",default="./aksharantar_sampled/tam/tam_train.csv", type=str,help="Path to training data")
    parser.add_argument("-tsp","--test_path",default="./aksharantar_sampled/tam/tam_test.csv", type=str,help="Path to testing data")
    parser.add_argument("-vlp","--valid_path",default="./aksharantar_sampled/tam/tam_valid.csv", type=str,help="Path to validation data")
    parser.add_argument("-emsz","--embedding_size",default=128, type=int,help="size of embedding.")
    parser.add_argument("-cl","--cell_type",default="GRU", type=str,choices = ['LSTM', 'RNN', 'GRU'],help="Which Cell type to use")
    parser.add_argument("-bd","--bidirectional",default="yes", type=str,choices = ['yes', 'no'],help="Whether to use bidirectional or not")
    parser.add_argument("-atn","--attention",default="yes", type=str,choices = ['yes', 'no'],help="Whether to use attention mechanism or not")
    
    args = parser.parse_args()

    project = args.wandb_project
    entity = args.wandb_entity
    epochs = args.epochs
    batch_size = args.batch_size 
    learning_rate = args.learning_rate
    dropout = args.dropout
    num_layers = args.num_layers
    hidden_layer_size = args.hidden_size
    
    TRAIN_PATH_a = args.train_path
    VALID_PATH_a = args.valid_path
    TEST_PATH_a = args.test_path
    
    if TRAIN_PATH_a=="":
        TRAIN_PATH_a = TRAIN_PATH
    if VALID_PATH_a=="":
        VALID_PATH_a = VALID_PATH
    if TEST_PATH_a=="":
        TEST_PATH_a = TEST_PATH
        
    
    embedding_size = args.embedding_size
    cell_type = args.cell_type
    
    bidirectional = args.bidirectional
    if bidirectional=="yes":
        bidirectional = True
    else:
        bidirectional = False
        
    attention = args.attention
    if attention=="yes":
        is_with_attention = True
    else:
        is_with_attention = False


    input_lang, output_lang, pairs = prepareData(INPUT_LANG_NAME, OUTPUT_LANG_NAME, TRAIN_PATH_a)
    _,_,train_dataloader = get_dataloader(batch_size,input_lang, output_lang,TRAIN_PATH_a)
    _,_,valid_dataloader = get_dataloader(batch_size,input_lang, output_lang,VALID_PATH_a)

    if is_with_attention:
        encoder = EncoderRNN(input_lang.n_chars, hidden_layer_size,embedding_size,dropout,num_layers,False,cell_type).to(DEVICE)
        decoder = AttnDecoderRNN(hidden_layer_size, output_lang.n_chars,embedding_size,num_layers,dropout,False,cell_type).to(DEVICE)
    else:  
        encoder = EncoderRNN(input_lang.n_chars, hidden_layer_size,embedding_size,dropout,num_layers,bidirectional,cell_type).to(DEVICE)
        decoder = DecoderRNN(hidden_layer_size,embedding_size, output_lang.n_chars,num_layers,bidirectional,cell_type).to(DEVICE)

    
    wandb.init(project=project,entity=entity)
    
    run_name = "{}_batchsize{}_epochs{}_lr{}_hlsize{}_embdsize{}_numlayer{}_dropout{}_bidirectional{}_with_attn{}".format(cell_type, batch_size, epochs, learning_rate, hidden_layer_size, embedding_size,
                            num_layers,dropout,bidirectional,attention)
    wandb.run.name = run_name
    train_model(train_dataloader,valid_dataloader, encoder, decoder,output_lang, epochs,  learning_rate=learning_rate) 
    
if __name__=="__main__":
    wandb.login()
    run_model()
    wandb.finish()
