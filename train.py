import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))

# stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
'''
The nn.CrossEntropyLoss() function is a loss function commonly used for classification problems.
The loss function is calculated between the model output and the real labels for the training data.
The model output is a probability vector for each class, then the loss function compares them
probabilities with true labels to determine how much the model output deviates from the truth label.

The torch.optim.Adam optimizer is a specific implementation of the Adam optimization algorithm.
Adam is a gradient based optimization algorithm which is used to update the weights of the
model during the training process. Adam uses a moving average of gradients to fit
dynamically the step size during the workout, which makes it efficient compared to others
optimization algorithms. The model.parameters() argument indicates that Adam should be used for optimization
the model weights and lr denotes the learning rate, which controls the size of the step used during
update model weights.'''

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        '''
        Queste righe di codice implementano l'algoritmo di backpropagation, che viene utilizzato per aggiornare 
        i pesi del modello durante il processo di addestramento. La funzione optimizer.zero_grad() azzera i gradienti
        accumulati durante l'ultimo passo di backpropagation. Questo è importante perché i gradienti vengono
        accumulati durante l'addestramento e, se non vengono azzerati, possono diventare molto grandi e causare 
        problemi di stabilità durante l'addestramento.

        La funzione loss.backward() calcola i gradienti della funzione di perdita rispetto ai pesi del modello.
        Questi gradienti vengono utilizzati dall'ottimizzatore per aggiornare i pesi del modello durante il passo
        successivo.Infine, la funzione optimizer.step() utilizza i gradienti calcolati durante il passo di
        backpropagation per aggiornare i pesi del modello. L'ottimizzatore utilizza questi gradienti per muovere
        i pesi del modello in una direzione che dovrebbe ridurre la funzione di perdita.'''
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')