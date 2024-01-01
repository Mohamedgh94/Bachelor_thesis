""" import torch
import torch.nn as nn
import torch.nn.functional as F

from embedding import EmbeddingLayer 
from encoder import EncoderLayer
from IMUDataset import IMUDataset



class GatedTransformer(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, d_ff, num_layers,num_person_ids , num_ages ,  num_heights, num_weights, num_genders):
        super(GatedTransformer, self).__init__()
        self.person_ids_classifier = nn.Linear(d_model, num_person_ids)
        self.age_classifier = nn.Linear(d_model, num_ages)
        self.height_classifier = nn.Linear(d_model, num_heights)
        self.weights_classifier = nn.Linear(d_model, num_weights)
        self.gender_classifier = nn.Linear(d_model,num_genders)
        self.embedding = nn.Linear(input_dim , d_model )
        print(f'self.embedding.weight.shape',self.embedding.weight.shape)
        self.encoders = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.classifier = nn.Linear(d_model, 5)  # 5 outputs for person, age, height, weight, and gender

    def forward(self, x):  # Unindented to match with __init__
        #print(f"Input shape: {x.shape}")  # Debugging line
        #print("Weight shape of embedding layer:", self.embedding.weight.shape)
        x = self.embedding(x)
        #print(f"After embedding shape: {x.shape}")  # Debugging line
        x = x.unsqueeze(0)  # Introduce a sequence length dimension of 1
        #print(f"After unsqueeze shape: {x.shape}")  # Debugging line
        for encoder in self.encoders:
            x = encoder(x)
            #print(f"After encoder shape: {x.shape}")  # Debugging line
        x = x.squeeze(0)  # Remove the sequence length dimension
        # print(f"After squeeze shape: {x.shape}")  # Debugging line
        # x = self.classifier(x)
        #print(f"Output shape: {x.shape}")  # Debugging line
        person_id_output = F.log_softmax(self.person_ids_classifier(x), dim=1)
        age_output = F.log_softmax(self.age_classifier(x), dim=1)
        height_output = F.log_softmax(self.height_classifier(x), dim=1)
        weight_output = F.log_softmax(self.weights_classifier(x), dim=1)
        gender_output = F.log_softmax(self.gender_classifier(x), dim=1)
        outputs= {
            'person_id': person_id_output,
            'age': age_output,
            'height': height_output,
            'weight': weight_output,
            'gender' : gender_output
        }
        return outputs
 """
import torch
import torch.nn as nn
import torch.nn.functional as F

from embedding import EmbeddingLayer 
from encoder import EncoderLayer

class GatedTransformer(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, d_ff, num_layers,
                 num_person_ids, num_ages, num_heights, num_weights, num_genders):
        super(GatedTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        #self.encoders = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        # self.person_ids_classifier = nn.Linear(d_model, num_person_ids)
        self.age_classifier = nn.Linear(d_model, num_ages)
        self.height_classifier = nn.Linear(d_model, num_heights)
        self.weight_classifier = nn.Linear(d_model, num_weights)
        self.gender_classifier = nn.Linear(d_model, num_genders)

        self.encoders = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.gate = nn.Linear(num_genders+num_person_ids, d_model)
    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(0)  # Introduce a sequence length dimension of 1

        for encoder in self.encoders:
            x = encoder(x)

        x = x.squeeze(0)  # Remove the sequence length dimension

        # Output raw logits for classification tasks
        # person_id_logits = self.person_ids_classifier(x)
        age_logits = self.age_classifier(x)
        height_logits = self.height_classifier(x)
        weight_logits = self.weight_classifier(x)
        gender_logits = self.gender_classifier(x)

        
        outputs = {
            'age': age_logits,
            'height': height_logits,
            'weight': weight_logits,
            'gender': gender_logits
        }
        return outputs
