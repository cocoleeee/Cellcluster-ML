import torch
import pandas as pd
from model.loss import M1_loss, M2_loss
from model.net import FullNet
from model.utils import init_data_loaders
import numpy as np
import matplotlib.pyplot as plt

class scSemiAE:
    def __init__(self, params, labeled_data, pretrain_data, hid_dim_1=500, hid_dim_2=50, p_drop=0.1):
        """
        Parameters
		----------
        params: 
            parameters of the scSemiAE model
        labeled_data: 
            labeled dataset. 
        pretrain_data: 
            dataset for pretraining. 
        hid_dim_1: int
            dimension in the first layer of the network (default: 500)
        hid_dim_2: int
            dimension in the second layer of the network (default: 50)
        p_drop: float
            dropout probability (default: 0.1)
        """
        # prepare data
        train_load, pretrain_load = init_data_loaders(labeled_data, pretrain_data, params.pretrain_batch)
        self.train_loader = train_load
        self.pretrain_loader = pretrain_load

        # prepare network
        x_dim = self.pretrain_loader.dataset.get_dim()
        self.init_model(x_dim, hid_dim_1, hid_dim_2, p_drop, params.device)

        # prepare parameters
        self.device = params.device
        self.epochs = params.epochs
        self.epochs_pretrain = params.epochs_pretrain

        self.lr = params.learning_rate
        self.lr_gamma = params.lr_scheduler_gamma
        self.step_size = params.lr_scheduler_step

        self.Lambda = params.Lambda


    def init_model(self, x_dim, hid_dim_1, hid_dim_2, p_drop, device):
        """
        Initialize the model.

        Parameters
		----------
        x_dim: int 
            imput dimension
        hid_dim_1: int
            dimension in the first layer of the network
        hid_dim_2: int
            dimension in the second layer of the network
        p_drop: float
            dropout probability
        device: "cpu" or "cuda"
        """
        self.model = FullNet(x_dim, hid_dim_1, hid_dim_2, p_drop = p_drop).to(device)
        
    def pretrain(self, optim, plot_flag=False):
        """
        Pretraining the model.
        
        Parameters
		----------
        optim:
            an optimizer for pretraining stages
        plot_flag: bool
            Whether to show the loss function of pretraining stages
        """

        print("# pretraining:")
        loss_M1 = []
        for _ in range(self.epochs_pretrain):
            loss_batch = 0
            for idx, batch in enumerate(self.pretrain_loader):
                x, _, _ = batch
                x = x.to(self.device)
                _, decoded = self.model(x)
                loss = M1_loss(decoded, x)
                loss_batch += loss.item()
                optim.zero_grad()
                loss.backward()
                optim.step()
            idx += 1 # idx starts at 0 and need to add 1 
            loss_M1.append(loss_batch / idx)
        if plot_flag:
            i = np.arange(0, self.epochs_pretrain, 1)
            plt.plot(i, loss_M1, color='red')
            plt.savefig('M1.jpg')
            plt.show()


    def train(self, pretrain_flag = False, plot_flag = False):
        """
        Training the model.
        
        Parameters
		----------
        pretrain_flag: bool
            Whether to pretrain only
        plot_flag: bool
            Whether to show the loss function
        
        Returns
		----------
        embeddings: dataframe
            low dimensional representations for cells
        """

        # pretraining stage
        optim_pretrain = torch.optim.Adam(params=list(self.model.parameters()), lr=self.lr)
        self.pretrain(optim_pretrain, plot_flag)
        print("pretraining is over!")

        # prepare the parameters
        lr = self.lr
        Lambda = self.Lambda
        epochs = 0 if pretrain_flag else self.epochs
        
        # fine-tuning stage
        optim = torch.optim.Adam(params = list(self.model.encoder.parameters()), lr=lr) #only optimize for encoder
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                               gamma=self.lr_gamma,
                                               step_size=self.step_size) # adjust the learing rate

        print("fine-tuning:")
        loss_M2 = []
        loss_L1 = []
        loss_L2 = []
        for epoch in range(epochs):
            M2, L1, L2 = 0, 0, 0
            for idx, batch in enumerate(self.train_loader):
                x, y, _ = batch
                x, y = x.to(self.device), y.to(self.device)
                encoded, _ = self.model(x)
                loss, L1_loss, L2_loss = M2_loss(encoded, y, Lambda)
                M2 += loss.item()
                L1 += L1_loss.item()
                L2 += L2_loss.item()
                optim.zero_grad()
                loss.backward()
                optim.step()
                lr_scheduler.step()
            idx += 1
            loss_M2.append(M2 / idx)
            loss_L1.append(L1 / idx)
            loss_L2.append(L2 / idx)

        if epochs > 0 and plot_flag:
            i = np.arange(0, epochs, 1)
            plt.plot(i, loss_M2, color='red', label="M2")
            plt.plot(i, loss_L1, color='blue', label='L1')
            plt.plot(i, loss_L2, color='green', label='L2')
            plt.legend(loc=0, ncol=1)
            plt.savefig('M2.jpg')
            plt.show()
        
        embeddings = self.compute_embeddings()
        return embeddings

    def compute_embeddings(self):
        """
        Saving embeddings from the dataset
        Returns
		----------
        embeddings: dataframe
            low dimensional representations for cells
        """

        embeddings_all = []

        for _, batch in enumerate(self.pretrain_loader):
            x, y, cells = batch
            x = x.to(self.device)
            encoded, _ = self.model(x)
            embeddings_all.append(pd.DataFrame(encoded.cpu().detach().numpy(), index = cells))
        
        # correspondding with labels
        embeddings = pd.concat(embeddings_all, axis=0).sort_index()

        return embeddings