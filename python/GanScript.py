import argparse
import os
import glob
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision

import scipy.io as io
import GanModelBuilder

from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule

import dataset_utils

os.makedirs("generated", exist_ok=True)

class ImageFolderDataset(Dataset):
    def __init__(self, imgs, targets):
        self.data = imgs
        self.targets = targets
    def __getitem__(self, index):
        x = self.data[index]
        x = np.fromfile(x, dtype=np.float32).reshape(3, 64, 64)
        y = self.targets[index]
        return x, y
    
    def __len__(self):
        return len(self.data)

class ImageMatlabDataset(Dataset):
    def __init__(self, imgs, targets):
        self.data = imgs
        self.targets = targets
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y
    
    def __len__(self):
        return len(self.data)
 

class LitAutoEncoder(LightningModule):
    def __init__(self, generator, discriminator, cond_loss, cond_lambda, described_species_labels, 
                 save_weights_path, lr_g=2e-5, lr_d=2e-4, b1=0.0, b2=0.999):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.cond_loss = cond_loss
        self.cond_lambda = cond_lambda
        self.described_species_labels = described_species_labels

        self.lr_g = lr_g
        self.lr_d = lr_d
        self.b1 = b1
        self.b2 = b2

        self.save_weights_path = save_weights_path

        self.automatic_optimization = False

    def forward(self, x):
        return self.generator(x)
    
    def on_predict_epoch_start(self):
        self.discriminator.eval()

    @torch.no_grad()
    def predict_step(self, batch):
        real_images, real_classes = batch
        disc_dict = self.discriminator.extract_features(real_images.to(self.device), real_classes.to(self.device)) 
        features_torch = disc_dict['feature']
        return features_torch.cpu().numpy()
    
    def training_step(self, batch):
        real_images, real_classes = batch
        batch_size = real_images.shape[0]
        
        self.generator.train()
        self.discriminator.train()

        optimizer_g, optimizer_d = self.optimizers()
        self.toggle_optimizer(optimizer_g)

        #TRAIN DISCRIMINATOR
        for _ in range(2):
            self.toggle_optimizer(optimizer_d)

            #use discriminator on real images
            real_dict = self.discriminator(real_images, real_classes)

            #generate and then use discriminator on fake images
            random_classes = torch.tensor(self.described_species_labels[np.random.randint(0, len(self.described_species_labels), batch_size)], device=self.device)
            t = self.generator(torch.randn(batch_size, 100, device=self.device, requires_grad=True), random_classes, eval=True)
            fake_dict = self.discriminator(t, random_classes)
            #Compute the two losses
            dis_acml_loss = GanModelBuilder.d_hinge(real_dict["adv_output"], fake_dict["adv_output"])
            real_cond_loss = self.cond_loss(**real_dict)
            dis_acml_loss += self.cond_lambda * real_cond_loss
            self.manual_backward(dis_acml_loss)
            optimizer_d.step()
            optimizer_d.zero_grad()
            self.untoggle_optimizer(optimizer_d)
        
        #TRAIN GENERATOR
        optimizer_g.zero_grad()
        random_classes = torch.tensor(self.described_species_labels[np.random.randint(0, len(self.described_species_labels), batch_size)],device=self.device)
        t = self.generator(torch.randn(batch_size,100).to(self.device), random_classes, eval = True)
        fake_dict = self.discriminator(t,random_classes)
        gen_acml_loss = GanModelBuilder.g_hinge(fake_dict["adv_output"])
        fake_cond_loss = self.cond_loss(**fake_dict)
        gen_acml_loss += self.cond_lambda * fake_cond_loss
        self.manual_backward(gen_acml_loss)
        optimizer_g.step()
        self.untoggle_optimizer(optimizer_g)

        self.log("gen_loss", gen_acml_loss, batch_size=batch_size)
        self.log("dis_loss", dis_acml_loss, batch_size=batch_size)

        total_loss = gen_acml_loss + dis_acml_loss

        return total_loss
    
    @torch.no_grad()
    def on_train_epoch_end(self):
        fixed_latent = torch.randn(100,100).to(self.device) # sample noise
        t = self.generator(fixed_latent, torch.tensor(np.arange(100)).to(self.device), eval = True)

        t = dataset_utils.denorm(t)
        p = torchvision.transforms.functional.to_pil_image(torchvision.utils.make_grid(t))
        p.save(f"generated/gan_training_epoch{self.current_epoch}.jpg")

        print(f"Saving model weights at {self.save_weights_path}")
        optimizer_g, optimizer_d = self.optimizers()
        
        torch.save({
                    'epoch': self.current_epoch,
                    'model_state_dict': self.generator.state_dict(),
                    'optimizer_state_dict': optimizer_g.state_dict(),
                    }, self.save_weights_path + "generator.pt")
        
        torch.save({
                    'epoch': self.current_epoch,
                    'model_state_dict': self.discriminator.state_dict(),
                    'optimizer_state_dict': optimizer_d.state_dict(),
                    }, self.save_weights_path + "discriminator.pt")

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(),     lr=self.lr_g, betas=(self.b1, self.b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_d, betas=(self.b1, self.b2))
        return [opt_g, opt_d], []

class Main:

    def __init__(self):

        parser = argparse.ArgumentParser(description="Train or extract features from DNA")
    
        # Define arguments
        parser.add_argument("-t", "--train" ,action="store_true", help="Train the model")
        parser.add_argument("-f", "--features" ,action="store_true", help="Extract the features")
        parser.add_argument("-e","--epochs", type=int, default=12, help="Number of epochs (default: 12 )")
        parser.add_argument("-b","--batch", type=int, default=16, help="Batch size (default: 16)")
        parser.add_argument("--train-on-val", action="store_true", help="Add this argument if you want to train on both training and validation set")
        parser.add_argument("--dataset-path", type=str, default="matlab_dataset/insect_dataset.mat", help="Path to the dataset for training")
        parser.add_argument("--save-weights-path", type=str, default="checkpoints/ImageGANWeights", help="Path to where to save the weights of the model after traininggt")
        parser.add_argument("--read-weights-path", type=str, default="checkpoints/ImageGANWeights", help="Path to where to read the weights of the model to extract features")

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
            self.ImageDataset = ImageMatlabDataset
            matlab_dataset = io.loadmat(dataset_path)

            self.all_images      = torch.tensor(matlab_dataset['all_images'])
            # self.all_dnas        = torch.tensor(matlab_dataset['all_dnas'])
            self.all_labels      = torch.tensor(matlab_dataset['all_labels']).squeeze()      - 1
            self.train_loc       = torch.tensor(matlab_dataset['train_loc']).squeeze()       - 1
            self.val_seen_loc    = torch.tensor(matlab_dataset['val_seen_loc']).squeeze()    - 1
            self.val_unseen_loc  = torch.tensor(matlab_dataset['val_unseen_loc']).squeeze()  - 1
            self.test_seen_loc   = torch.tensor(matlab_dataset['test_seen_loc']).squeeze()   - 1
            self.test_unseen_loc = torch.tensor(matlab_dataset['test_unseen_loc']).squeeze() - 1
            self.species2genus   = torch.tensor(matlab_dataset['species2genus']) - 1

            self.described_labels_train    = matlab_dataset['described_species_labels_train'].squeeze()    - 1
            self.described_labels_trainval = matlab_dataset['described_species_labels_trainval'].squeeze() - 1
        else:
            assert os.path.isdir(dataset_path), "Dataset path must be a folder or matlab .mat file"
            self.ImageDataset = ImageFolderDataset
            self.all_images = glob.glob(f"{dataset_path}/imgs/*.np"); self.all_images.sort()
            self.all_dnas   = glob.glob(f"{dataset_path}/dnas/*.np"); self.all_dnas.sort()
            
            self.all_labels      = torch.Tensor(np.fromfile(f'{dataset_path}/all_labels.np',      dtype=np.int64)).long()
            self.train_loc       = torch.Tensor(np.fromfile(f'{dataset_path}/train_loc.np',       dtype=np.int64)).long()
            self.val_seen_loc    = torch.Tensor(np.fromfile(f'{dataset_path}/val_seen_loc.np',    dtype=np.int64)).long()
            self.val_unseen_loc  = torch.Tensor(np.fromfile(f'{dataset_path}/val_unseen_loc.np',  dtype=np.int64)).long()
            self.test_seen_loc   = torch.Tensor(np.fromfile(f'{dataset_path}/test_seen_loc.np',   dtype=np.int64)).long()
            self.test_unseen_loc = torch.Tensor(np.fromfile(f'{dataset_path}/test_unseen_loc.np', dtype=np.int64)).long()
            self.species2genus   = torch.Tensor(np.fromfile(f'{dataset_path}/species2genus.np',   dtype=np.int64)).long()

            self.described_labels_train    = np.fromfile(f'{dataset_path}/described_labels_train.np',    dtype=np.int64)
            self.described_labels_trainval = np.fromfile(f'{dataset_path}/described_labels_trainval.np', dtype=np.int64)


    def train_execution(self):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        batch_size = self.args.batch
        # Read dataset
        
        if self.args.train_on_val:
            self.train_loc = torch.cat((self.train_loc, self.val_loc))
            self.val_loc = self.test_loc
            self.described_species_labels = self.described_labels_trainval
            print("Training on both train and val set")

        else:
            self.described_species_labels = self.described_labels_train

        train_d = self.ImageDataset(np.array(self.all_images)[self.train_loc], self.all_labels[self.train_loc])
        train_loader = DataLoader(train_d, batch_size=batch_size, shuffle=True, num_workers=2)

        (discriminator, generator) = GanModelBuilder.model_builder() 
            
        cond_loss = GanModelBuilder.Data2DataCrossEntropyLoss(self.n_classes, 0.5,0.98, device)
        cond_lambda = 1 

        model = LitAutoEncoder(generator, discriminator, cond_loss, cond_lambda, self.described_species_labels, self.args.save_weights_path)

        trainer = Trainer(
            accelerator="auto",
            devices=1,
            max_epochs=5,
        )
        trainer.fit(model, train_loader)

        
        
    def feature_execution(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        (discriminator, generator) = GanModelBuilder.model_builder()
        
        if os.path.isdir(self.args.read_weights_path):
            path_d = os.path.join(self.args.read_weights_path, "discriminator.pt")
        else:
            path_d = self.args.read_weights_path

        print("Loading discriminator weights from", path_d)
        print(torch.load(path_d).keys())
        discriminator.load_state_dict(torch.load(path_d)['model_state_dict'])

        discriminator.to(device)

        train_d = self.ImageDataset(np.array(self.all_images), self.all_labels)
        train_loader = DataLoader(train_d, batch_size=self.args.batch, shuffle=False, num_workers=2)
        
        model = LitAutoEncoder(generator, discriminator, None, None, None, self.args.save_weights_path)

        trainer = Trainer(
            accelerator="auto",
            devices=1,
        )
        all_image_features = trainer.predict(model, train_loader)
        all_image_features = np.concatenate(all_image_features)
        print("all_image_features shape", all_image_features.shape)

        features_dataset = dict()
        features_dataset['all_image_features_gan'] = all_image_features 
        print("Saving features to all_image_features_gan.mat")
        io.savemat('all_image_features_gan.mat',features_dataset)
   
if __name__ == "__main__":
    Main()
