import torch
import torch.nn as nn
import torch.nn.functional as F
from PositionalEncoding import PositionalEncoding
from EncoderLayer import EncoderLayer
import torch.utils.checkpoint as checkpoint
from IMUDataset import num_person_ids,num_genders

class GatedTransformer(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, d_ff, num_layers,
                 num_person_ids, num_ages, num_heights, num_weights, num_genders):
        super(GatedTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.encoders = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])

        # Task-specific layers
        self.person_ids_classifier = nn.Linear(d_model, num_person_ids)
        self.age_regressor = nn.Linear(d_model, 1)
        self.height_regressor = nn.Linear(d_model, 1)
        self.weight_regressor = nn.Linear(d_model, 1)
        self.gender_classifier = nn.Linear(d_model, num_genders)

        # Gating mechanism
        self.gate = nn.Linear(num_genders+num_person_ids, d_model)
    
    def forward(self, x):
        x = self.embedding(x)
        
        x = self.positional_encoding(x)
        
        # Apply each encoder in the list
        for encoder in self.encoders:
            x = encoder(x)
            
        last_token_output = x[:, -1, :]

        person_id_logits = self.person_ids_classifier(last_token_output)
        gender_logits = self.gender_classifier(last_token_output)
        # Split the output for each task
       
        
        age_output = self.age_regressor(x[:, -1, :]).squeeze().float()  # Squeezing to shape [batch_size]
        height_output = self.height_regressor(x[:, -1, :]).squeeze().float()  # Squeezing to shape [batch_size]
        weight_output = self.weight_regressor(x[:, -1, :]).squeeze().float()  # Squeezing to shape [batch_size]
        
        
        # Apply gating mechanism
        #print(f'person_id_logits.shape: {person_id_logits.shape}')
        #print(f'gender_logits.shape : {gender_logits.shape}')
        #gated_outputs = self.gate(torch.cat((person_id_logits, gender_logits), dim=-1))
        gated_outputs = self.gate(torch.cat((person_id_logits, gender_logits), dim=1))
        
        outputs = {
            'person_id': person_id_logits,
            'age': age_output,
            'height': height_output,
            'weight': weight_output,
            'gender': gender_logits,
            'gated': gated_outputs
        }
        return outputs