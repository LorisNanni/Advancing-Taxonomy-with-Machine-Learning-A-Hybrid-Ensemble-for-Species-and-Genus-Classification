import argparse
import os
import glob
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import scipy.io as io
from DnaModel import TinyModel

from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule

class DNAFolderDataset(Dataset):
    def __init__(self, imgs, targets):
        self.data    = imgs
        self.targets = targets
    def __getitem__(self, index):
        x = self.data[index]
        x = np.fromfile(x, dtype=np.float64).reshape(658, 5)
        x = torch.tensor(x).float().unsqueeze(0)
        y = self.targets[index]
        return x, y
    
    def __len__(self):
        return len(self.data)

class DNAMatlabDataset(Dataset):
    def __init__(self, imgs, targets):
        self.data = imgs
        self.targets = targets
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y
    
    def __len__(self):
        return len(self.data)

class LightningModel(LightningModule):
    def __init__(self, model, criterion, save_weights_path):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.save_weights_path = save_weights_path

        self.running_train_corrects = self.running_valid_corrects = 0
        self.tot_train = self.tot_valid = 0


    def training_step(self, batch, batch_idx):
        dnas, labels = batch
        # import ipdb; ipdb.set_trace()
        predicted_labels = self.model(dnas)
        loss = self.criterion(predicted_labels, labels)
        self.log('train_loss', loss)
        _, preds = torch.max(predicted_labels, 1)
        self.running_train_corrects += torch.sum(preds == labels.data)
        self.tot_train += len(labels)
        return loss
    
    @torch.no_grad()
    def on_train_epoch_end(self):
        epoch_train_acc = self.running_train_corrects.double() / self.tot_train
        print(f"Epoch {self.current_epoch} train accuracy: {epoch_train_acc}")
        self.tot_train = self.running_train_corrects = 0 # reset
        optimizer = self.optimizers()

        torch.save({
                    'epoch':self.current_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, os.path.join(self.save_weights_path, "dnamodel.pt"))

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        dnas, labels = batch
        predicted_labels = self.model(dnas)
        loss = self.criterion(predicted_labels, labels)
        self.log('val_loss', loss)
        _, preds = torch.max(predicted_labels, 1)
        self.running_valid_corrects += torch.sum(preds == labels.data)
        self.tot_valid += len(labels)
        return preds
    
    @torch.no_grad()
    def on_validation_epoch_end(self):
        epoch_valid_acc = self.running_valid_corrects.double() / self.tot_valid
        print(f"Epoch {self.current_epoch} valid accuracy: {epoch_valid_acc}")
        self.tot_valid = self.running_valid_corrects = 0 # reset

    @torch.no_grad()
    def predict_step(self, batch):
        dnas, _ = batch
        fts = self.model.feature_extract(dnas)
        return fts.cpu().numpy()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)

class Main:
    def __init__(self):

        parser = argparse.ArgumentParser(description="Train or extract features from DNA")
        
        # Define arguments
        parser.add_argument("-t", "--train" ,action="store_true", help="Train the model")
        parser.add_argument("-f", "--features" ,action="store_true", help="Extract the features")
        parser.add_argument("-e","--epochs", type=int, default=50, help="Number of epochs (default: 50)")
        parser.add_argument("-b","--batch", type=int, default=32, help="Batch size (default: 32)")
        parser.add_argument("--train-on-val", action="store_true", help="Add this argument if you want to train on both training and validation set")
        parser.add_argument("--dataset-path", type=str, default="matlab_dataset/insect_dataset.mat", help="Path to the dataset for training")
        parser.add_argument("--save-weights-path", type=str, default="checkpoints/DnaCNNWeights.pt", help="Path to where to save the weights of the model after traininggt")
        parser.add_argument("--read-weights-path", type=str, default="checkpoints/DnaCNNWeights.pt", help="Path to where to read the weights of the model to extract features")

        # Parse arguments
        self.args = parser.parse_args()

        self.read_data()
        self.val_loc  = torch.cat((self.val_seen_loc,  self.val_unseen_loc))
        self.test_loc = torch.cat((self.test_seen_loc, self.test_unseen_loc))

        self.n_classes = self.all_labels.max()+1
        print("using n_classes", self.n_classes)

        if self.args.train:
            self.train_execution()
        elif self.args.features:
            self.feature_execution()
        else:
            parser.print_help()
            exit()
    
    def read_data(self):
        dataset_path = self.args.dataset_path
        if dataset_path[-4:] == ".mat":
            self.DNADataset = DNAMatlabDataset
            matlab_dataset = io.loadmat(dataset_path)

            # self.all_images      = torch.tensor(matlab_dataset['all_images'])
            self.all_dnas        = torch.tensor(matlab_dataset['all_dnas'])
            self.all_labels      = torch.tensor(matlab_dataset['all_labels']).squeeze()      - 1
            self.train_loc       = torch.tensor(matlab_dataset['train_loc']).squeeze()       - 1
            self.val_seen_loc    = torch.tensor(matlab_dataset['val_seen_loc']).squeeze()    - 1
            self.val_unseen_loc  = torch.tensor(matlab_dataset['val_unseen_loc']).squeeze()  - 1
            self.test_seen_loc   = torch.tensor(matlab_dataset['test_seen_loc']).squeeze()   - 1
            self.test_unseen_loc = torch.tensor(matlab_dataset['test_unseen_loc']).squeeze() - 1

        else:
            assert os.path.isdir(dataset_path), "Dataset path must be a folder or matlab .mat file"
            self.DNADataset = DNAFolderDataset
            # self.all_images = glob.glob(f"{dataset_path}/imgs/*.np"); self.all_images.sort()
            self.all_dnas   = glob.glob(f"{dataset_path}/dnas/*.np"); self.all_dnas.sort()
            
            self.all_labels      = torch.Tensor(np.fromfile(f'{dataset_path}/all_labels.np',      dtype=np.int64)).long()
            self.train_loc       = torch.Tensor(np.fromfile(f'{dataset_path}/train_loc.np',       dtype=np.int64)).long()
            self.val_seen_loc    = torch.Tensor(np.fromfile(f'{dataset_path}/val_seen_loc.np',    dtype=np.int64)).long()
            self.val_unseen_loc  = torch.Tensor(np.fromfile(f'{dataset_path}/val_unseen_loc.np',  dtype=np.int64)).long()
            self.test_seen_loc   = torch.Tensor(np.fromfile(f'{dataset_path}/test_seen_loc.np',   dtype=np.int64)).long()
            self.test_unseen_loc = torch.Tensor(np.fromfile(f'{dataset_path}/test_unseen_loc.np', dtype=np.int64)).long()


    def train_execution(self):

        if self.args.train_on_val:
            self.train_loc = torch.cat((self.train_loc, self.val_loc))
            self.val_loc = self.test_loc
            print("Training on both train and val set")

        batch_size = self.args.batch

        d_train = self.DNADataset(np.array(self.all_dnas)[self.train_loc], self.all_labels[self.train_loc])
        d_val   = self.DNADataset(np.array(self.all_dnas)[self.val_loc],   self.all_labels[self.val_loc]  )
        d_test  = self.DNADataset(np.array(self.all_dnas)[self.test_loc],  self.all_labels[self.test_loc] )
        
        dataloader_train = DataLoader(d_train, batch_size=batch_size, shuffle=True)
        dataloader_val   = DataLoader(d_val,   batch_size=batch_size, shuffle=False)
        # dataloader_test  = DataLoader(d_test,  batch_size=batch_size, shuffle=False)
            
        print(f"Training for {self.args.epochs} epochs")
            
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        tinymodel = TinyModel()
        tinymodel.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        
        lightningModel = LightningModel(tinymodel, criterion, self.args.save_weights_path)
        trainer = Trainer(
            accelerator='auto',
            devices=1,
            max_epochs=50,
        )
        trainer.fit(lightningModel, dataloader_train, dataloader_val)
        


    def feature_execution(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tinymodel = TinyModel()
        if os.path.isdir(self.args.read_weights_path):
            path_d = os.path.join(self.args.read_weights_path, "dnamodel.pt")
        else:
            path_d = self.args.read_weights_path
        
        state_dict = torch.load(path_d)
        tinymodel.load_state_dict(state_dict['model_state_dict'])
        tinymodel.to(device)

        alldna_d = self.DNADataset(np.array(self.all_dnas), self.all_labels)
        alldna_loader = DataLoader(alldna_d, batch_size=self.args.batch, shuffle=False, num_workers=2)
        
        model = LightningModel(tinymodel, None, None)

        trainer = Trainer(
            accelerator="auto",
            devices=1,
        )
        all_dna_features = trainer.predict(model, alldna_loader)

        all_dna_features = np.concatenate(all_dna_features)
        print("all_dna_features shape", all_dna_features.shape)
        
        features_dataset = dict()
        features_dataset['all_dna_features_cnn_new'] = all_dna_features 
        io.savemat('all_dna_features_cnn_new.mat',features_dataset)
    
if __name__ == "__main__":
    Main()
