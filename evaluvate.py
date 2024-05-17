import torch
from load_data import WordFromTensor

def evaluate(dataloader, encoder, decoder, criterion, output_lang):
    total_loss = 0
    total_correct = 0
    total_samples = 0
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        for data in dataloader:
            input_tensor, target_tensor = data

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)


            # compute loss
            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )
            total_loss += loss.item()

            # Compute accuracy
            _, predicted_indices = decoder_outputs.max(dim=-1)
            #print(predicted_indices.shape)
            # correct = (predicted_indices == target_tensor).all(dim=1).sum().item()
            correct = 0
            for i in range(predicted_indices.shape[0]):
                if WordFromTensor(output_lang,predicted_indices[i])==WordFromTensor(output_lang,target_tensor[i]):
                    correct += 1

            total_correct += correct
            total_samples += input_tensor.size(0)


    # Calculate average loss and accuracy
    avg_loss = total_loss / len(dataloader)
    accuracy = (total_correct / total_samples) * 100

    return avg_loss, accuracy