import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import pickle

import torchvision.transforms as transforms
from models import Encoder, EncodeVAE_Encoder
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}
        self.data_folder = data_folder
        # Captions per image
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        encPath = self.data_folder + '/TrainResnet101Features/' + str(i // self.cpi) + '.p'
        encVAEPath = self.data_folder + '/TrainResnet101Features/VAE_' + str(i // self.cpi) + '.p'
        encodeImage = pickle.load( open( encPath, "rb" ) )
        encodeVAEImage = pickle.load( open( encVAEPath, "rb" ) )
        encodeImageT = torch.FloatTensor(encodeImage)
        encodeVAEImageT = torch.FloatTensor(encodeVAEImage)

        caption = torch.LongTensor(self.captions[i])
        caplen = torch.LongTensor([self.caplens[i]])

        if self.split is 'TRAIN':
            return caption, caplen, encodeImageT
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return caption, caplen, all_captions, encodeImageT, encodeVAEImageT

    def __len__(self):
        return self.dataset_size


#to preprocess feature extraction
def GetResnet101Features():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_folder = 'C:/Users/paoca/Documents/UVA PHD/NLP/PROJECT/UnnecesaryDataFolder'  # folder with data files saved by create_input_files.py
    data_name = 'coco_5_cap_per_img_5_min_word_freq'

    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=5, shuffle=False, pin_memory=True)

    with torch.no_grad():
        encoder = Encoder()
        encoder.fine_tune(False)

        emb_dim = 512
        decoder_dim = 512
        encoderVae_encoder = EncodeVAE_Encoder(embed_dim=emb_dim,
                                                decoder_dim=decoder_dim,
                                                vocab_size=len(word_map))
        encoderVae_encoder.fine_tune(False)

        encoder.eval()
        encoderVae_encoder.eval()

        encoder = encoder.to(device)
        encoderVae_encoder = encoderVae_encoder.to(device)

        for i, (imgs, caps, caplens) in enumerate(train_loader):
            if i % 100 == 0:
                print (i)

            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            res = encoder(imgs)
            h = encoderVae_encoder(imgs, caps, caplens)

            pickle.dump( res[0].cpu().numpy(), open( "C:/Users/paoca/Documents/UVA PHD/NLP/PROJECT/UnnecesaryDataFolder/TrainResnet101Features/" + str(i) + ".p", "wb" ) )
            pickle.dump( h[0].cpu().numpy(), open( "C:/Users/paoca/Documents/UVA PHD/NLP/PROJECT/UnnecesaryDataFolder/TrainResnet101Features/VAE_" + str(i) + ".p", "wb" ) )


if __name__ == '__main__':
    GetResnet101Features()
