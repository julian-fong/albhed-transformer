import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tokenizer import get_tokenizer
import torch

path = "../data/translations.csv"
df = pd.read_csv(path)
tokenizer = get_tokenizer()

class AlbhedDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.eos_id = 50256  # GPT-2 EOS token ID
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        albhed_text = self.dataframe.iloc[idx]['albhed']
        english_text = self.dataframe.iloc[idx]['english']

        albhed_tokens = self.tokenizer.encode(albhed_text)
        english_tokens = self.tokenizer.encode(english_text)
        
        return {
            'encoder_input_ids': torch.tensor(english_tokens, dtype=torch.long),
            'decoder_input_ids': torch.tensor([self.eos_id] + albhed_tokens, dtype=torch.long),
            'labels': torch.tensor(albhed_tokens + [self.eos_id], dtype=torch.long),
            'albhed_text': albhed_text,
            'english_text': english_text
        }

import torch
from torch.nn.utils.rnn import pad_sequence

def custom_collate_fn(batch):
    encoder_sequences = [item['encoder_input_ids'] for item in batch]
    decoder_sequences = [item['decoder_input_ids'] for item in batch]
    labels_sequences = [item['labels'] for item in batch]
    
    encoder_sizes = [arr.shape[0] for arr in encoder_sequences] # list of length of each encoder sentence
    decoder_sizes = [arr.shape[0] for arr in decoder_sequences] # list of length of each decoder sentence
    
    largest_encoder_sequence_length = max(encoder_sizes) # maximum length of encoder sentences in the batch
    largest_decoder_sequence_length = max(decoder_sizes) # maximum length of decoder sentences in the batch
    
    encoder_attention_mask = []
    for length_of_encoder_sentence in encoder_sizes:
        ones = torch.ones(length_of_encoder_sentence, dtype=torch.int64)
        zeros = torch.zeros(largest_encoder_sequence_length - length_of_encoder_sentence, dtype=torch.int64)
        sentence_mask = torch.cat((ones, zeros), dim=0).unsqueeze(0)  # Add batch dimension
        encoder_attention_mask.append(sentence_mask)
    encoder_attention_mask = torch.cat(encoder_attention_mask, dim = 0)
    
    decoder_attention_mask = []
    for length_of_decoder_sentence in decoder_sizes:
        ones = torch.ones(length_of_decoder_sentence, dtype=torch.int64)
        zeros = torch.zeros(largest_decoder_sequence_length - length_of_decoder_sentence, dtype=torch.int64)
        sentence_mask = torch.cat((ones, zeros), dim=0).unsqueeze(0)  # Add batch dimension
        decoder_attention_mask.append(sentence_mask)
    decoder_attention_mask = torch.cat(decoder_attention_mask, dim = 0)


    # Pad the input sequences to the maximum length in the current batch
    # padding_value can be customized
    encoder_padded_sequences = pad_sequence(encoder_sequences, batch_first=True, padding_value=50257, padding_side = 'right')
    decoder_padded_sequences = pad_sequence(decoder_sequences, batch_first=True, padding_value=50257, padding_side = 'right')
    labels_padded_sequences = pad_sequence(labels_sequences, batch_first=True, padding_value=50257, padding_side = 'right')
    
    new_batch = {}

    new_batch['encoder_input_ids'] = encoder_padded_sequences
    new_batch['decoder_input_ids'] = decoder_padded_sequences
    new_batch['labels'] = labels_padded_sequences
    new_batch['encoder_attention_mask'] = encoder_attention_mask
    new_batch['decoder_attention_mask'] = decoder_attention_mask

    return new_batch

def get_albhed_dataset():
    """
    Returns the Albhed dataset.
    """
    dataset = AlbhedDataset(dataframe=df, tokenizer=tokenizer)
    return dataset

def get_albhed_dataloader(batch_size=8, shuffle=True):
    """
    Returns the DataLoader for the Albhed dataset.
    """
    dataset = get_albhed_dataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate_fn)
    return dataloader