from torch.utils.data import Dataset, DataLoader, Sampler
import itertools
import os
from glob import glob
import numpy as np 
import random 
from torch.autograd import Variable
import torch 

Tensor = torch.Tensor


class InputDataset(Dataset): 
        def __init__(self, root, instrument_list, unaligned=True):
            """
            """
            self.root = root
            self.instrument_list = instrument_list
            self.nclass = len(self.instrument_list) # number of instruments
            self.unaligned= unaligned
            self.size = []
            # 
            self.path_info = {}
            for inst in instrument_list:
                self.path_info[inst] = {'paths': [], 'size': 0}
                mag_path = os.path.join(root, inst, 'mag')
                phase_path = os.path.join(root, inst, 'phase')
                npy_list_mag = glob(f'{mag_path}/*.npy')
                npy_list_phase = glob(f'{phase_path}/*.npy')
                if len(npy_list_mag) != len(npy_list_phase):
                    raise RuntimeError(f'Number of magnitude files not equal number of phase files for insturment {inst}')
                self.path_info[inst]['paths'] += npy_list_mag
                self.path_info[inst]['size'] += len(npy_list_mag)
                self.size.append(len(npy_list_mag))
            
            self.cumsum_size = np.array(self.size).cumsum()
            
            # compute length 
            length = 0
            # 2 files from 1 instrument. 
            for inst in self.instrument_list:
                length += self.path_info[inst]['size'] 
            self.homo_threshold = length
            # 2 files from 2 instruments
            self.hete_thresholds = []
            self.pairs = []
            for pair in itertools.combinations(range(self.nclass),2):
                a, b = pair
                len_a = self.path_info[self.instrument_list[a]]['size']
                len_b = self.path_info[self.instrument_list[b]]['size']
                length += max(len_a, len_b)
                self.hete_thresholds.append(length)
                self.pairs.append([a,b])
            self.length = length
                
        def __len__(self):
            return self.length
        
        def __getitem__(self, idx):
            if idx < self.homo_threshold: # 2 iles from 1 instrument
                inst_ind = self.cumsum_size.searchsorted(idx, side='right') # get index of the instrument
                ind_in_inst = idx - self.cumsum_size[inst_ind-1] if inst_ind > 0 else idx
                file_name_A = self.path_info[self.instrument_list[inst_ind]]['paths'][ind_in_inst%self.size[inst_ind]]
                file_A = np.load(file_name_A)
                if self.unaligned:
                    file_name_B = self.path_info[self.instrument_list[inst_ind]]['paths'][random.randint(0, self.size[inst_ind]-1)]
                    file_B = np.load(file_name_B)
                else:
                    file_name_B = self.path_info[self.instrument_list[inst_ind]]['paths'][(ind_in_inst+1)%self.size[inst_ind]]
                    file_B = np.load(file_name)
                a_ind, b_ind = inst_ind, inst_ind
                
            else: # 2 files from different instruments
                pair_ind = np.array(self.hete_thresholds).searchsorted(idx, side='right')
                a_ind, b_ind = self.pairs[pair_ind]
                ind_in_inst = idx - self.hete_thresholds[pair_ind-1] if pair_ind > 0 else idx - self.homo_threshold
                file_name_A = self.path_info[self.instrument_list[a_ind]]['paths'][ind_in_inst % self.size[a_ind]]
                file_A = np.load(file_name_A)
                if self.unaligned:
                    file_name_B = self.path_info[self.instrument_list[b_ind]]['paths'][random.randint(0,self.size[b_ind]-1)]
                    file_B = np.load(file_name_B)
                else:    
                    file_name_B = self.path_info[self.instrument_list[b_ind]]['paths'][ind_in_inst % self.size[b_ind]]
                    file_B = np.load(file_name_B)
                    
            file_A = np.expand_dims(file_A, axis=0) 
            file_B = np.expand_dims(file_B, axis=0) 
            return {'A': file_A, 'B': file_B, 'lab_A': a_ind, 'lab_B': b_ind}



if __name__ == "__main__":
    data = InputDataset(root='./data/spectrogram', instrument_list=['piano', 'harpsichord'])
    loader = DataLoader(data, batch_size=19, shuffle=True)
    for i, batch in enumerate(loader):
        print(Variable(batch["A"].type(Tensor)).size())