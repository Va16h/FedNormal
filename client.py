import gc
import pickle
import logging

import torch
import torch.nn as nn
from distoptim.FedDNA import FedDNA
from distoptim.FedProx import FedProx
from distoptim.FedNova import FedNova
from distoptim.Scaffold import Scaffold

from importlib import reload # reload 

import numpy as np
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class Client(object):
    """Class for client object having its own (private) data and resources to train a model.

    Participating client has its own dataset which are usually non-IID compared to other clients.
    Each client only communicates with the center server with its trained parameters or globally aggregated parameters.

    Attributes:
        id: Integer indicating client's id.
        data: torch.utils.data.Dataset instance containing local data.
        device: Training machine indicator (e.g. "cpu", "cuda").
        __model: torch.nn instance as a local model.
    """
    def __init__(self, client_id, local_data, device):
        """Client object is initiated by the center server."""
        self.id = client_id
        self.data = local_data
        self.device = device
        self.__model = None
        self.controls = None

    @property
    def model(self):
        """Local model getter for parameter aggregation."""
        return self.__model

    @model.setter
    def model(self, model):
        """Local model setter for passing globally aggregated model parameters."""
        self.__model = model

    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.data)

    def setup(self, **client_config):
        """Set up common configuration of each client; called by center server."""
        self.dataloader = DataLoader(self.data, batch_size=client_config["batch_size"], shuffle=True)
        self.local_epoch = client_config["num_local_epochs"]
        self.criterion = client_config["criterion"]
        self.optimizer = client_config["optimizer"]
        self.optim_config = client_config["optim_config"]
        self.lr = self.optim_config["lr"]



    def client_update(self, server_controls, server_model):
        """Update local model using local dataset."""
        self.model.train()
        self.model.to(self.device)

        # self.optimizer is str (the name of optimizer)
        # this is the object of optimizer
        self.optimizer_object = eval(self.optimizer)(self.model.parameters(), **self.optim_config)

        if (self.controls == None) :
            self.controls = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]        

        local_steps = 0

        for e in range(self.local_epoch):
            for data, labels in self.dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
  
                self.optimizer_object.zero_grad()
                outputs = self.model(data)
                loss = eval(self.criterion)()(outputs, labels)

                loss.backward()

                # pass server, client controls here
                self.optimizer_object.step(server_controls, self.controls)

                local_steps += 1                    

                if self.device == "cuda": torch.cuda.empty_cache()               
        
        self.delta_controls = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad] 
        new_controls = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]           

        for idx, server_p, client_p in zip(range(len(self.controls)), server_model.parameters(), self.model.parameters()): 
            new_controls[idx].data = self.controls[idx].data - server_controls[idx].data + (1 / (local_steps * self.lr)) * (server_p.data - client_p.data)

        # get controls differences
        for idx in range(len(self.controls)):
            self.delta_controls[idx].data = new_controls[idx].data - self.controls[idx].data
            self.controls[idx].data = new_controls[idx].data        

        self.client_params = torch.nn.utils.parameters_to_vector(self.model.parameters())
        self.c_i = self.client_params - torch.nn.utils.parameters_to_vector(server_model.parameters())

        self.model.to("cpu")


    def client_evaluate(self):
        """Evaluate local model using local dataset (same as training set for convenience)."""
        self.model.eval()
        self.model.to(self.device)

        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels in self.dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = self.model(data)
                test_loss += eval(self.criterion)()(outputs, labels).item()
                
                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()

                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")

        test_loss = test_loss / len(self.dataloader)
        test_accuracy = correct / len(self.data)

        message = f"\t[Client {str(self.id).zfill(4)}] ...finished evaluation!\
            \n\t=> Test loss: {test_loss:.4f}\
            \n\t=> Test accuracy: {100. * test_accuracy:.2f}%\n"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        return test_loss, test_accuracy

def angle_dot(a, b):
    dot_product = torch.dot(a, b)
    prod_of_norms = torch.linalg.norm(a) * torch.linalg.norm(b)
    return dot_product / prod_of_norms        