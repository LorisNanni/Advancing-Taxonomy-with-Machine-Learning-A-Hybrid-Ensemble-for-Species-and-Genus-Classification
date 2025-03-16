import numpy as np
import pandas as pd 
from typing import List
from typing import Tuple 
from sklearn.model_selection import train_test_split
import random 
import torch
from torchvision.utils import save_image
import os 
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

def one_hot_encoding(nucleotide: str, seq_len=658) -> np.ndarray:
    # Cutting the sequence if it is longer than a pre-defined value seq_len
    if len(nucleotide) > seq_len:
        nucleotide = nucleotide[:seq_len]
    # Encoding
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    sequence = [mapping[i] if i in mapping else 4 for i in nucleotide]
    encoded_sequence = np.eye(5)[sequence]
    # Padding if the sequence is smaller than a pre-defined value seq_len
    if len(encoded_sequence) < seq_len:
        padding = np.zeros((seq_len - len(encoded_sequence), 5))
        encoded_sequence = np.concatenate((encoded_sequence, padding))
    return encoded_sequence


def get_imgs_bold_id(image_dataset,df):
  img2dna = dict()
  not_found_images = []
  for i, row in df.iterrows():
      urls = row['image_urls'].split('|')
      species_name = row['species_name'].replace(' ','_')
      #if len(urls) >1 and row['species_name'] == 'Drosophila affinis':
      for url in urls:
          image_name_csv ='image_dataset/'+species_name+'/'+url[url.rfind('/')+1:]
          #print(image_name_csv)
          trovato = False
          for img in image_dataset.imgs:
              if img[0] == image_name_csv:
                  img2dna[img[0]]= row['processid']
                  trovato = True
                  break
          if not trovato:
              not_found_images.append(image_name_csv)
  return img2dna

def data_split(df, test_ratio,drop_labels = False, random_state = 42):
    test = []
    genus_count = df.groupby('genus_name')['species_name'].nunique()
    
    for genus_name in genus_count.index:
        number_undescribed_species = genus_count[genus_name]//3
        species = list(df.loc[df['genus_name']==genus_name]['species_name'].unique())
        undescribed_species = random.sample(species,number_undescribed_species)
        test = test+undescribed_species

    df_remaining = df.loc[~df.species_name.isin(test)]
    df_undescribed = df.loc[df.species_name.isin(test)]
    
    y = df_remaining['species_name']
    if drop_labels:
        X = df_remaining.drop(columns=['species_name','genus_name','processid','image_urls'])
        X_undescribed = df_undescribed.drop(columns=['species_name','genus_name','processid','image_urls'])
    else:
        X = df_remaining
        X_undescribed = df_undescribed
    y_undescribed = df_undescribed['species_name']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=random_state)
    y_test = pd.concat([y_test,y_undescribed])
    X_test = pd.concat([X_test,X_undescribed])
    return X_train, X_test, y_train, y_test

def image_filenames_from_df(df: pd.core.frame.DataFrame) -> Tuple[List[str],List[str]]:
    filenames : List[str] = []
    boldids : List[str] = []

    for i,row in df.iterrows():
        urls = row['image_urls']
        boldid = row['processid']
        for url in urls.split('|'):
            species_name = row['species_name'].replace(' ','_')
            image_filename = 'image_dataset/'+species_name+'/'+url[url.rfind('/')+1:]
            filenames.append(image_filename)
            boldids.append(boldid)
    return filenames,boldids

def image_splits_from_df(X_train, X_validation,X_test,image_dataset):
    (train_filenames ,train_bolds) = image_filenames_from_df(X_train)
    (val_filenames ,val_bolds) = image_filenames_from_df(X_validation)
    (test_filenames ,test_bolds) = image_filenames_from_df(X_test)
    train_indices : List[int] = []
    val_indices : List[int] = []
    test_indices : List[int] = []
    boldids : List[str] = []
    assert len(train_filenames) == len(train_bolds)
    assert len(val_filenames) == len(val_bolds)
    assert len(test_filenames) == len(test_bolds)
    for i, (filename,label) in enumerate(image_dataset.imgs):
        if filename in train_filenames:
            train_indices.append(i)
            filename_index = train_filenames.index(filename)
            boldids.append(train_bolds[filename_index])
        elif filename in val_filenames:
            val_indices.append(i)
            filename_index = val_filenames.index(filename)
            boldids.append(val_bolds[filename_index])
        elif filename in test_filenames:
            test_indices.append(i)
            filename_index = test_filenames.index(filename)
            boldids.append(test_bolds[filename_index])
        else:
            raise Exception(f"Exception: Filename {filename} isn't in any of the splits")
    return train_indices,val_indices,test_indices,boldids

def get_dataset(image_path:str, csv_path:str,batch_size:int, shuffle_loaders:bool = False):
    df = pd.read_csv(csv_path,index_col=0)
    imsize = 64
    tform = transforms.Compose([transforms.Resize((imsize,imsize)),
                                transforms.PILToTensor(),
                                transforms.ConvertImageDtype(torch.float),
                                transforms.Normalize(0.5,0.5)])
    image_dataset = torchvision.datasets.ImageFolder(image_path, transform=tform)

    img2dna = get_imgs_bold_id(image_dataset,df)

    nucleotides = df[['nucleotide','species_name','genus_name','processid','image_urls']]
    colonna_dna = df.loc[:,"nucleotide"]
    nucleotides.loc[:,'nucleotide'] = colonna_dna.apply(one_hot_encoding)
    random.seed(42)
    X_train_val, X_test, y_train_val, y_test = data_split(nucleotides,0.2,random_state=42)
    train_data = X_train_val
    train_data['species_name'] = y_train_val
    X_train, X_validation, y_train, y_validation = data_split(train_data,0.2,drop_labels=False,random_state=42)
    train_indices, val_indices, test_indices = image_splits_from_df(X_train,X_validation,X_test,image_dataset)
     
    class WholeDataset(Dataset):
        def __init__(self, data, transform=None):
            self.data = data
            self.targets = data.targets            
        def __getitem__(self, index):
            x = self.data[index][0]
            y = self.targets[index]
            return x, y
        
        def __len__(self):
            return len(self.data)
            
    whole_dataset = WholeDataset(image_dataset)
    n_classes = np.unique(whole_dataset.targets).shape[0]
    train_imgs = torch.utils.data.Subset(whole_dataset, train_indices)
    val_imgs = torch.utils.data.Subset(whole_dataset, val_indices)
    train_val_imgs = torch.utils.data.Subset(whole_dataset, train_indices+val_indices)
    test_imgs = torch.utils.data.Subset(whole_dataset, test_indices)

    train_loader = torch.utils.data.DataLoader(train_imgs, batch_size=batch_size,shuffle=shuffle_loaders, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_imgs, batch_size=batch_size,shuffle=shuffle_loaders, num_workers=2)
    train_val_loader = torch.utils.data.DataLoader(train_val_imgs, batch_size=batch_size,shuffle=shuffle_loaders, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_imgs, batch_size=batch_size,shuffle=shuffle_loaders, num_workers=2)
    dataloaders = {"train":train_loader,"val":val_loader,"test":test_loader,'train_val':train_val_loader}
    dataset_sizes = {'train': len(train_imgs.indices), 'val':len(val_imgs.indices),'test':len(test_imgs.indices),'train_val':len(train_val_imgs.indices)}
    
    described_species_labels = np.array([image_dataset.targets[i] for i in train_indices])
    described_species_labels = np.unique(described_species_labels)
    
    return dataloaders,dataset_sizes,described_species_labels,n_classes
def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

class Save_samples_params():
    def __init__(self,latent_tensors,sample_random_classes,sample_dir):
        self.latent_tensors = latent_tensors
        self.sample_random_classes = sample_random_classes
        self.sample_dir = sample_dir
class Fit_params():
    def __init__(self,discriminator_optimizer,generator_optimizer,discriminator,generator,dataloaders,device,writer,batch_size,n_classes,latent_size,described_species_labels):
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer
        self.discriminator = discriminator
        self.generator = generator
        self.dataloaders = dataloaders
        self.device = device
        self.writer = writer
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.latent_size = latent_size
        self.described_species_labels = described_species_labels
def save_samples(index, save_p : Save_samples_params, generator ,writer,show=True):
    with torch.no_grad():
        fake_images = generator(save_p.latent_tensors,save_p.sample_random_classes)
        fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
        save_image(denorm(fake_images), os.path.join(save_p.sample_dir, fake_fname), nrow=8)
        writer.add_image('sample image',denorm(fake_images[0]),global_step=index)
        print('Saving', fake_fname)
        if show:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))
def denorm(img_tensors):
    return img_tensors * 0.5 + 0.5

def species_label_to_genus_label(df : pd.DataFrame, image_dataset):
    genuses = df['genus_name'].unique()
    species = df['species_name'].unique()
    img2dna = get_imgs_bold_id(image_dataset,df)
    specie2genus = {}
    for specie in species:
        label_specie = image_dataset.class_to_idx[specie.replace(" ","_")]
        genus = df.loc[df['species_name']==specie]['genus_name'].unique()[0]
        label_genus = np.where(genuses == genus)[0][0]
        #specie2genus[specie] = genus
        specie2genus[label_specie] = label_genus
    return specie2genus

def top2_choice(image_val_labels, val_predicted_labels, described_species_labels, val_predicted_probs, species2genus):
    tprs = []
    fprs = []
    correct_genus_rate = []
    correct_species_rate = []
    for t in range(0,100,1):
        entropy_threshold = t/100.0
        n_undescribed_samples = 0
        n_described_samples = 0
        n_correct_undescribed_samples = 0
        n_correct_described_samples = 0
        n_correct_genus = 0 
        n_correct_species = 0 
        for i in range(len(image_val_labels)):
            
            label_best_specie = val_predicted_labels[i]
           
            assert(val_predicted_labels[i]==val_predicted_probs[i].argmax())
            genus_of_best_species = species2genus[label_best_specie.item()]
            
            sorted_probs = np.sort(val_predicted_probs[i])
            sorted_probs = sorted_probs[::-1]
            
            prob_diff = abs(sorted_probs[0] - sorted_probs[1])
            
            if image_val_labels[i].item() in described_species_labels:
                #tn
                n_described_samples +=1
                if prob_diff >= entropy_threshold:
                    n_correct_described_samples+=1
                    if label_best_specie == image_val_labels[i]:
                        n_correct_species+=1
            else:
                #tp
                n_undescribed_samples+=1
                if prob_diff < entropy_threshold:
                    n_correct_undescribed_samples+=1
                    real_genus = species2genus[image_val_labels[i].item()]
                    predicted_genus = genus_of_best_species
                    if real_genus == predicted_genus:
                        n_correct_genus+=1
            
                
        tprs.append(n_correct_undescribed_samples/n_undescribed_samples) # TPR = recall = sensitivity
        fprs.append(1-n_correct_described_samples/n_described_samples) # 1-TNR = 1 - specificity
        correct_genus_rate.append(n_correct_genus/n_undescribed_samples)
        correct_species_rate.append(n_correct_species/n_described_samples)

        return tprs, fprs, correct_genus_rate, correct_species_rate