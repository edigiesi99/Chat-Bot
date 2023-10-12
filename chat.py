import random
import json
import numpy as np

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "ChatBot"

class try_class():

    def initiate(self):
        self.HyperFabric = 0
        self.HyperIndy = 0
        self.BitCoin = 0
        self.LiteCoin = 0
        self.Nexledger = 0
        self.Algorand = 0
        self.Quadrans = 0
        self.ETH = 0
        self.Polygon = 0
        self.Quorum = 0
        self.Flow = 0

    def __init__(self):
        self.initiate()

    def get_response(self,msg):

        sentence = tokenize(msg)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        # restituisce il massimo valore presente in ogni riga della matrice output
        # e la posizione di tale valore. La posizione viene assegnata alla variabile predicted, 
        # mentre il massimo viene ignorato poiché non è di nostro interesse"
        _, predicted = torch.max(output, dim=1) 

        tag = tags[predicted.item()]
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        # se tags è composto da tre elementi, "saluto", "domanda" e "informazione", e il vettore restituito da torch.softmax è [0.3, 0.4, 0.3], significa
        # che il modello ha una probabilità del 30% di classificare il messaggio come "saluto", una probabilità del 40% di classificarlo come "domanda"
        # e una probabilità del 30% di classificarlo come "informazione". La probabilità corrispondente all'etichetta ottenuta viene quindi assegnata alla variabile prob
        
        if prob.item() > 0.75:
            
            for intent in intents['intents']:

                if tag == intent["tag"]:

                    if tag == "Benvenuto":
                        return (f"{bot_name}: {random.choice(intent['responses'])}")

                    if tag == "Fine":
                        return(f"{bot_name}: {random.choice(intent['responses'])}")

                    if tag == "thanks":
                        return(f"{bot_name}: {random.choice(intent['responses'])}")

                    if tag == "Inizio blockchain migliore":                    
                        return (f"{bot_name}: {intent['responses'][0]}")
                        
                    if tag == "trasparenza":
                        for i in sentence:
                            if i.isnumeric():
                                if int(i) == 1:
                                    pass
                                if int(i) == 2:
                                    pass
                                if int(i) == 3:
                                    self.HyperFabric += 1
                                if int(i) == 4:
                                    self.HyperIndy += 1
                                    self.Nexledger += 1
                                    self.Algorand += 1
                                    self.Quadrans += 1
                                    self.Quorum += 1
                                    self.Flow += 1
                                    
                                if int(i) == 5:
                                    self.BitCoin += 1
                                    self.LiteCoin += 1
                                    self.ETH += 1
                                    self.Polygon += 1

                                    

                        return (f"{bot_name}: {intent['responses'][0]}")
                    
                    if tag == "sicurezza":
                        for i in sentence:
                            if i.isnumeric():
                                if int(i) == 1:
                                    pass
                                if int(i) == 2:
                                    pass
                                if int(i) == 3:
                                    self.Quorum += 1
                                    self.HyperIndy += 1
                                if int(i) == 4:
                                    self.HyperFabric += 1
                                    self.ETH += 1
                                    self.LiteCoin += 1
                                    self.Nexledger += 1
                                    self.Algorand += 1
                                    self.Flow += 1
                                if int(i) == 5:
                                    self.Polygon += 1
                                    self.Quadrans += 1
                                    self.BitCoin += 1

                                    
                        return (f"{bot_name}: {intent['responses'][0]}")
                 
                    if tag == "velocità":
                        for i in sentence:
                            if i.isnumeric():
                                if int(i) == 1:
                                    pass
                                if int(i) == 2:
                                    self.BitCoin += 1
                                if int(i) == 3:
                                    self.LiteCoin += 1
                                if int(i) == 4:
                                    self.HyperIndy += 1
                                    self.Algorand += 1
                                    self.Quadrans += 1
                                    self.ETH += 1
                                if int(i) == 5:
                                    self.HyperFabric += 1
                                    self.Nexledger += 1
                                    self.Polygon += 1
                                    self.Quorum += 1
                        return (f"{bot_name}: {intent['responses'][0]}")
                    
                    if tag == "automazione":

                        for i in sentence:
                            if i.isnumeric():
                                if int(i) == 1:
                                    pass
                                if int(i) == 2:
                                    self.BitCoin += 1
                                if int(i) == 3:
                                    self.Nexledger += 1
                                    self.Quadrans += 1
                                if int(i) == 4:
                                    self.HyperFabric += 1
                                    self.ETH += 1
                                if int(i) == 5:
                                    self.HyperIndy += 1
                                    self.Algorand += 1
                                    self.Polygon += 1
                                    self.Quorum += 1
                                    self.Flow += 1
                        return (f"{bot_name}: {intent['responses'][0]}")
                    
                    if tag == "linguaggio":
                        for i in sentence:
                            if i.isnumeric():
                                if int(i) == 1:
                                    pass
                                if int(i) == 2:
                                   self.Flow += 1
                                if int(i) == 3:
                                    self.Quadrans += 1
                                    self.ETH += 1
                                    self.Polygon += 1
                                if int(i) == 4:
                                   self.Algorand += 1
                                   self.Quorum += 1
                                if int(i) == 5:
                                    self.HyperFabric += 1

                        return (f"{bot_name}: {intent['responses'][0]}")
                    
                    if tag == "costi":
                        for i in sentence:
                            if i.isnumeric():
                                if int(i) == 1:
                                    self.HyperFabric += 1
                                    self.HyperIndy += 1
                                    self.Nexledger += 1
                                    self.Quorum += 1
                                if int(i) == 2:
                                    self.Algorand += 1
                                    self.LiteCoin += 1
                                    self.Flow += 1
                                if int(i) == 3:
                                    self.Polygon += 1
                                if int(i) == 4:
                                    self.ETH += 1
                                if int(i) == 5:
                                    self.BitCoin += 1
                                                 
                        return (f"{bot_name}: {intent['responses'][0]}")

                    if tag == "documentazione":
                        for i in sentence:
                            if i.isnumeric():
                                if int(i) == 1:
                                    pass
                                if int(i) == 2:
                                   self.HyperIndy += 1
                                   self.Algorand += 1
                                   self.Polygon += 1
                                if int(i) == 3:
                                    self.Nexledger += 1
                                if int(i) == 4:
                                    self.HyperFabric += 1
                                    self.LiteCoin += 1
                                    self.Quadrans += 1
                                    self.Quorum += 1
                                    self.Flow += 1
                                if int(i) == 5:
                                    self.BitCoin += 1
                                    self.ETH += 1
                                                 
                        return (f"{bot_name}: {intent['responses'][0]}")

                    if tag == "trovato":
                        print('For each blockchain the number of matches found is:')
                        print("HyperFabric" ,self.HyperFabric)
                        print("HyperIndy" ,self.HyperIndy) 
                        print("BitCoin" ,self.BitCoin) 
                        print("LiteCoin" ,self.LiteCoin) 
                        print("Nexledger" ,self.Nexledger) 
                        print("Algorand" ,self.Algorand) 
                        print("Quadrans" ,self.Quadrans) 
                        print("ETH" ,self.ETH) 
                        print("Polygon" ,self.Polygon)
                        print("Quorum" ,self.Quorum)
                        print("Flow" ,self.Flow)

                        cryptocurrencies = {
                            "HyperFabric": {"transparency" : "3/5", "speech_difficulties" :"5/5", "security" : "4/5", "automation" : "4/5", "speed" : "5/5","documentation" :"4/5","cost" :"1/5"},
                            "HyperIndy": {"transparency" : "4/5", "speech_difficulties" :"null", "security" : "3/5", "automation" : "5/5", "speed" : "4/5","documentation" :"2/5","cost" :"1/5"},
                            "BitCoin": {"transparency" : "5/5", "speech_difficulties" :"null", "security" : "5/5", "automation" : "2/5", "speed" : "2/5","documentation" :"5/5","cost" :"5/5"},
                            "LiteCoin": {"transparency" : "5/5", "speech_difficulties" :"null", "security" : "4/5", "automation" : "null", "speed" : "3/5","documentation" :"4/5","cost" :"2/5"},
                            "Nexledger": {"transparency" : "4/5", "speech_difficulties" :"null", "security" : "4/5", "automation" : "3/5", "speed" : "5/5","documentation" :"3/5","cost" :"1/5"},
                            "Algorand": {"transparency" : "4/5", "speech_difficulties" :"4/5", "security" : "4/5", "automation" : "5/5", "speed" : "4/5","documentation" :"2/5","cost" :"2/5"},
                            "Quadrans":{"transparency" : "4/5", "speech_difficulties" :"3/5", "security" : "5/5", "automation" : "3/5", "speed" : "4/5","documentation" :"4/5","cost" :"null"},
                            "ETH": {"transparency" : "5/5", "speech_difficulties" :"3/5", "security" : "4/5", "automation" : "4/5", "speed" : "4/5","documentation" :"5/5","cost" :"4/5"},
                            "Polygon": {"transparency" : "5/5", "speech_difficulties" :"3/5", "security" : "5/5", "automation" : "5/5", "speed" : "5/5","documentation" :"2/5","cost" :"3/5"},
                            "Quorum": {"transparency" : "4/5", "speech_difficulties" :"4/5", "security" : "3/5", "automation" : "5/5", "speed" : "5/5","documentation" :"4/5","cost" :"1/5"},
                            "Flow": {"transparency" : "4/5", "speech_difficulties" :"2/5", "security" : "4/5", "automation" : "5/5", "speed" : "null","documentation" :"4/5","cost" :"2/5"},
                        }
                  
                        values = np.array([self.HyperFabric, self.HyperIndy, self.BitCoin,self.LiteCoin,self.Nexledger,self.Algorand,self.Quadrans,self.ETH,self.Polygon,self.Quorum,self.Flow])
                        nomi = ["HyperFabric", "HyperIndy", "BitCoin","LiteCoin","Nexledger","Algorand","Quadrans","ETH","Polygon","Quorum","Flow"]

                        # Troviamo i valori massimi nella lista
                        max_value = np.max(values)

                        # Troviamo gli indici corrispondenti ai valori massimi
                        max_indices = [i for i, x in enumerate(values) if x == max_value]

                        max_index = np.argmax(values)

                        if len(max_indices) >= 2:

                            message = "I'm sorry but I found more than one match with the values provided, here are the results! 🤓\n"
                            max_names = [nomi[i] for i in max_indices]

                            # Costruiamo la stringa di output con i nomi e i valori delle criptovalute individuate
                            max_names_str = ""
                            self.initiate()

                            for name in max_names:
                                max_names_str += f"{name}: \n transparency={cryptocurrencies[name]['transparency']},\n speech_difficulties={cryptocurrencies[name]['speech_difficulties']},\n security={cryptocurrencies[name]['security']},\n automation={cryptocurrencies[name]['automation']},\n speed={cryptocurrencies[name]['speed']},\n documentation={cryptocurrencies[name]['documentation']},\n cost={cryptocurrencies[name]['cost']}\n \n \n"

                            return bot_name +":"+ message + "\n" + max_names_str


                        #HyperFabric
                        if max_index == 0:
                            self.initiate()
                            name = "HyperFabric"
                            return f"{name}: \n transparency={cryptocurrencies[name]['transparency']},\n speech_difficulties={cryptocurrencies[name]['speech_difficulties']},\n security={cryptocurrencies[name]['security']},\n automation={cryptocurrencies[name]['automation']},\n speed={cryptocurrencies[name]['speed']},\n documentation={cryptocurrencies[name]['documentation']},\n cost={cryptocurrencies[name]['cost']}\n \n \n"

                        #HyperIndy
                        if max_index == 1:
                            self.initiate()
                            name = "HyperIndy"
                            return f"{name}: \n transparency={cryptocurrencies[name]['transparency']},\n speech_difficulties={cryptocurrencies[name]['speech_difficulties']},\n security={cryptocurrencies[name]['security']},\n automation={cryptocurrencies[name]['automation']},\n speed={cryptocurrencies[name]['speed']},\n documentation={cryptocurrencies[name]['documentation']},\n cost={cryptocurrencies[name]['cost']}\n \n \n"
                          
                        #BitCoin
                        if max_index == 2:
                            self.initiate()
                            name = "BitCoin"
                            return f"{name}: \n transparency={cryptocurrencies[name]['transparency']},\n speech_difficulties={cryptocurrencies[name]['speech_difficulties']},\n security={cryptocurrencies[name]['security']},\n automation={cryptocurrencies[name]['automation']},\n speed={cryptocurrencies[name]['speed']},\n documentation={cryptocurrencies[name]['documentation']},\n cost={cryptocurrencies[name]['cost']}\n \n \n"
                        
                        #LiteCoin
                        if max_index == 3:
                            self.initiate()
                            name = "LiteCoin"
                            return f"{name}: \n transparency={cryptocurrencies[name]['transparency']},\n speech_difficulties={cryptocurrencies[name]['speech_difficulties']},\n security={cryptocurrencies[name]['security']},\n automation={cryptocurrencies[name]['automation']},\n speed={cryptocurrencies[name]['speed']},\n documentation={cryptocurrencies[name]['documentation']},\n cost={cryptocurrencies[name]['cost']}\n \n \n"
                        
                        #Nexledger
                        if max_index == 4:
                            self.initiate()
                            name = "Nexledger"
                            return f"{name}: \n transparency={cryptocurrencies[name]['transparency']},\n speech_difficulties={cryptocurrencies[name]['speech_difficulties']},\n security={cryptocurrencies[name]['security']},\n automation={cryptocurrencies[name]['automation']},\n speed={cryptocurrencies[name]['speed']},\n documentation={cryptocurrencies[name]['documentation']},\n cost={cryptocurrencies[name]['cost']}\n \n \n"
                        
                        #Algorand
                        if max_index == 5:
                            self.initiate()
                            name = "Algorand"
                            return f"{name}: \n transparency={cryptocurrencies[name]['transparency']},\n speech_difficulties={cryptocurrencies[name]['speech_difficulties']},\n security={cryptocurrencies[name]['security']},\n automation={cryptocurrencies[name]['automation']},\n speed={cryptocurrencies[name]['speed']},\n documentation={cryptocurrencies[name]['documentation']},\n cost={cryptocurrencies[name]['cost']}\n \n \n"
                        
                        #Quadrans
                        if max_index == 6:
                            self.initiate()
                            name = "Quadrans"
                            return f"{name}: \n transparency={cryptocurrencies[name]['transparency']},\n speech_difficulties={cryptocurrencies[name]['speech_difficulties']},\n security={cryptocurrencies[name]['security']},\n automation={cryptocurrencies[name]['automation']},\n speed={cryptocurrencies[name]['speed']},\n documentation={cryptocurrencies[name]['documentation']},\n cost={cryptocurrencies[name]['cost']}\n \n \n"
                        
                        #ETH
                        if max_index == 7:
                            self.initiate()
                            name = "ETH"
                            return f"{name}: \n transparency={cryptocurrencies[name]['transparency']},\n speech_difficulties={cryptocurrencies[name]['speech_difficulties']},\n security={cryptocurrencies[name]['security']},\n automation={cryptocurrencies[name]['automation']},\n speed={cryptocurrencies[name]['speed']},\n documentation={cryptocurrencies[name]['documentation']},\n cost={cryptocurrencies[name]['cost']}\n \n \n"
                        
                        # Polygon
                        if max_index == 8:
                            self.initiate()
                            name = "Polygon"
                            return f"{name}: \n transparency={cryptocurrencies[name]['transparency']},\n speech_difficulties={cryptocurrencies[name]['speech_difficulties']},\n security={cryptocurrencies[name]['security']},\n automation={cryptocurrencies[name]['automation']},\n speed={cryptocurrencies[name]['speed']},\n documentation={cryptocurrencies[name]['documentation']},\n cost={cryptocurrencies[name]['cost']}\n \n \n"
                           
                        #Quorum
                        if max_index == 9:
                            self.initiate()
                            name = "Quorum"
                            return f"{name}: \n transparency={cryptocurrencies[name]['transparency']},\n speech_difficulties={cryptocurrencies[name]['speech_difficulties']},\n security={cryptocurrencies[name]['security']},\n automation={cryptocurrencies[name]['automation']},\n speed={cryptocurrencies[name]['speed']},\n documentation={cryptocurrencies[name]['documentation']},\n cost={cryptocurrencies[name]['cost']}\n \n \n"
                        
                        #Flow
                        if max_index == 10:
                            self.initiate()
                            name = "Flow"
                            return f"{name}: \n transparency={cryptocurrencies[name]['transparency']},\n speech_difficulties={cryptocurrencies[name]['speech_difficulties']},\n security={cryptocurrencies[name]['security']},\n automation={cryptocurrencies[name]['automation']},\n speed={cryptocurrencies[name]['speed']},\n documentation={cryptocurrencies[name]['documentation']},\n cost={cryptocurrencies[name]['cost']}\n \n \n"
                                                                                             
                        self.initiate()
                                            
        else:
            return("I didn't understand the question, try to rephrase it!😕")