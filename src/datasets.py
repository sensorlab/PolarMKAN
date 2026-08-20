import h5py
import numpy as np
from torch.utils.data import Dataset
import torch
import pickle
import pywt
from torchvision import transforms
import torchvision
import glob
import json

class OracleDataset(Dataset):
    def __init__(self, file: str, input_size: int = 256, devices=None, 
                 return_indices=False, type = 'train', modality = 'iq_const'):
        self.return_indices = return_indices
        self.modality = modality
        self.devices_map = [10,  4,  6,  0, 15,  1,  9,  2, 11,  3, 12,  8,  5,  7, 13, 14]

        
        with h5py.File(file) as f:
            self.data_i = np.array(f['data_i']).astype(np.float32)
            self.data_q = np.array(f['data_q']).astype(np.float32)
            self.labels = np.array(f['labels'][:,0]).astype(int)
            self.data_i_saved = self.data_i.reshape(-1,256)
            self.data_q_saved = self.data_q.reshape(-1,256)
            self.labels_saved = self.labels

        if type == 'validation':
            self.data_i = self.data_i.reshape(16,2, 2000,256)[:,:,-400:].reshape(-1,256)
            self.data_q = self.data_q.reshape(16,2, 2000,256)[:,:,-400:].reshape(-1,256)
            self.labels = self.labels.reshape(16,2, 2000)[:,:,-400:].reshape(-1)
        else:
            self.data_i = self.data_i.reshape(16,2, 2000,256)[:,:,:-400].reshape(-1,256)
            self.data_q = self.data_q.reshape(16,2, 2000,256)[:,:,:-400].reshape(-1,256)
            self.labels = self.labels.reshape(16,2, 2000)[:,:,:-400].reshape(-1)
            
        self.data_i = self.data_i[[i for i in range(len(self.labels)) if self.devices_map[self.labels[i]] in  devices]]
        self.data_q = self.data_q[[i for i in range(len(self.labels)) if self.devices_map[self.labels[i]] in  devices]]
        self.labels = self.labels[[i for i in range(len(self.labels)) if self.devices_map[self.labels[i]] in  devices]]
    
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        if self.modality == 'iq_const':
            i_samples = self.data_i[idx] / np.abs(self.data_i_saved[self.labels_saved == label]).max()
            q_samples = self.data_q[idx] / np.abs(self.data_q_saved[self.labels_saved == label]).max()

            i_samples = (i_samples + 1) / 2 * 100
            q_samples = (q_samples + 1) / 2 * 100
            
            i_indices = i_samples.astype(int)
            q_indices = q_samples.astype(int)
            i_indices = np.clip(i_indices, 0, 99)
            q_indices = np.clip(q_indices, 0, 99)
            sample = np.zeros((1,100,100))
            for j in range(len(i_samples)):
                sample[0][i_indices[j], q_indices[j]] += 1
            #sample = sample / sample.max()
            sample = sample[:,20:80,20:80].astype(np.float32)
        else:
            data_q = self.data_q[idx] /  np.abs(self.data_i_saved[self.labels_saved == label]).max()
            data_i = self.data_i[idx] /  np.abs(self.data_q_saved[self.labels_saved == label]).max()
            sample = np.stack([data_i, data_q],axis = 0)
            sample = sample
        
        return (sample, self.devices_map[label], idx) if self.return_indices else (sample, self.devices_map[label])

class LoRaDataset(Dataset):
    """
    Soruce:  LoRa Device Fingerprinting in the Wild: Disclosing RF Data-Driven Fingerprint Sensitivity to Deployment
Variability. IEEE Access, pp: 142893–142909, October 2021.
    """
    def __init__(self, filedir: str, input_size: int = 1024, devices=None, selected_days=None, 
                 return_indices=False, transform_to_2d=None):
        self.devices = devices
        self.selected_days = selected_days
        self.filedir = filedir
        self.transform_to_2d = transform_to_2d
        self.return_indices = return_indices
        self.slice_length = input_size * 2
        self.total_days = len(selected_days)
        self.total_devices = len(devices)
        self.total_transmissions = 1
        self.transmission_length = 400000
        self.total_len = (self.total_days * self.total_devices * self.total_transmissions * self.transmission_length) // self.slice_length

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        cur_div = self.total_devices * self.total_transmissions * self.transmission_length // self.slice_length
        day = idx // cur_div
        idx %= cur_div

        cur_div = self.total_transmissions * self.transmission_length // self.slice_length
        device = idx // cur_div
        idx %= cur_div

        cur_div = self.transmission_length // self.slice_length
        transmission = idx // cur_div
        slice_idx = idx % cur_div

        with open(f"{self.filedir}/Day_{self.selected_days[day]}/device_{self.devices[device]}/trans_{transmission+1}.dat", 'rb') as f:
            data = np.frombuffer(f.read(), np.float32)
            data = data[slice_idx * self.slice_length: (slice_idx + 1) * self.slice_length]
            data_i = data[::2]
            data_q = data[1::2]
            sample = np.stack([data_i, data_q], axis=0)

        device = self.devices[device]
        return (sample, device, idx) if self.return_indices else (sample, device)



class WiSig_Dataset_ManySig(Dataset):
    """
    Soure: S. Hanna, S. Karunaratne, and D. Cabric, “WiSig: A Large-Scale WiFi Signal Dataset for Receiver and Channel Agnostic RF Fingerprinting,” IEEE Access, vol. 10, pp. 22808–22818, 2022, doi: 10.1109/ACCESS.2022.3154790.
    """
    def __init__(self, file: str, devices=None, days=None, selected_receivers=None, return_indices=False, transform_to_2d=None, train_test_split = False, type = 'train', polars_c = False, k_fold_samples = 0, polars = False, cfo_compensate = False):
        with open(file, 'rb') as f:
            self.data = pickle.load(f)

        self.transform_to_2d = transform_to_2d
        self.return_indices = return_indices
        self.polars = polars
        self.cfo_compensate = cfo_compensate
        self.devices = devices or list(range(len(self.data['tx_list'])))
        self.selected_receivers = selected_receivers or list(range(len(self.data['rx_list'])))
        self.selected_days = days or list(range(len(self.data['capture_date_list'])))
        self.polars_c = polars_c

        if type == 'validation' and train_test_split:
            self.data_ordered = [
                (sample, i)
                for i in self.devices
                for j in self.selected_receivers
                for m in self.selected_days
                for sample in self.data['data'][i][j][m][1][-200:]
            ]
            
            
        elif type == 'train' and train_test_split:
            self.data_ordered = [
                (sample, i)
                for i in self.devices
                for j in self.selected_receivers
                for m in self.selected_days
                for sample in self.data['data'][i][j][m][1][:-200]
            ]
            
        else:
            self.data_ordered = [
                (sample, i)
                for i in self.devices
                for j in self.selected_receivers
                for m in self.selected_days
                for sample in self.data['data'][i][j][m][1]
            ]

    def __len__(self):
        return len(self.data_ordered)


    def __getitem__(self, idx):
        sample, tx = self.data_ordered[idx]
        sample = sample.T


        i_comp, q_comp = sample[0], sample[1]

        if self.cfo_compensate:
            # Blind per-burst bulk-CFO removal: estimate mean phase slope and de-rotate.
            x = i_comp.astype(np.float64) + 1j * q_comp.astype(np.float64)
            nidx = np.arange(x.shape[-1])
            cfo = np.mean(np.angle(x[1:] * np.conj(x[:-1]))) / (2 * np.pi)
            x = x * np.exp(-1j * 2 * np.pi * cfo * nidx)
            i_comp, q_comp = np.real(x), np.imag(x)
            sample = np.stack([i_comp, q_comp])

        # Cartesian -> polar (magnitude, phase in [0, 2pi)).
        magnitude = np.sqrt(i_comp ** 2 + q_comp ** 2)
        phase = np.arctan2(q_comp, i_comp)
        phase = (phase + 2 * np.pi) % (2 * np.pi)

        if self.polars:
            # Scale roughly into [0, 1] for both channels.
            sample = np.stack([magnitude / 1.5, phase / (2 * np.pi)])
        
        sample = sample.astype(np.float32)

        return (sample, tx, idx) if self.return_indices else (sample, tx)


class WiSig_Dataset_ManyTx(Dataset):
    """
    Soure: S. Hanna, S. Karunaratne, and D. Cabric, “WiSig: A Large-Scale WiFi Signal Dataset for Receiver and Channel Agnostic RF Fingerprinting,” IEEE Access, vol. 10, pp. 22808–22818, 2022, doi: 10.1109/ACCESS.2022.3154790.
    """
    def __init__(self, file: str, devices=None, days=None, selected_receivers=None, return_indices=False, transform_to_2d=None, train_test_split = False, type = 'train', k_fold_samples = 0, polars = False, cfo_compensate = False):
        with open(file, 'rb') as f:
            self.data = pickle.load(f)

        self.transform_to_2d = transform_to_2d
        self.return_indices = return_indices
        self.polars = polars
        self.cfo_compensate = cfo_compensate
        self.selected_receivers = selected_receivers or list(range(len(self.data['rx_list'])))
        self.selected_days = days or list(range(len(self.data['capture_date_list'])))
        self.device_map = [ 17,  40, 107,  97,   2,  70, 133, 102, 136,  88,  46,  36,  34,
        62,   1, 149,  69, 145,   4, 127,  78,  81,  96,  68,  74, 142,
        45, 130, 106,  47,  79,  53,  59,  98,  20,  57, 137, 143,   5,
        105,  65, 112, 113,  54,  44, 117, 119, 148,  18,  52,  41,  19,
        51, 111,  38, 128,  12,   8, 147, 100,  95, 129,  99,   3,  77,
        123, 110,  73,   7,  93, 141,  63,  29,  13,  61,  43,  25,  71,
        132,  23,  82, 115,  66, 104, 103,  49,  85, 118,  76,  80,  24,
        32,  55,  28,  94, 144, 140,  16,  84,  75,   0,  91, 131,  11,
        21,  37, 122,  72,   9, 139, 126,   6,  30, 134,  48,  86, 146,
        90,  83,  42,  27, 138, 121, 125, 108,  87,  56, 135,  58, 120,
        109,  64, 114,  14,  10,  35, 101,  89,  39,  33,  22,  50,  26,
        116,  31,  60,  15,  92,  67, 124]
        
        devices = devices or list(range(len(self.data['tx_list'])))
        self.devices = [i for i in range(150) if i in devices]
        
        if type == 'validation' and train_test_split:

            cur_len = len(self.data['data'][0][0][0][0])
            
            self.data_ordered = [
                (sample, i)
                for i in self.devices
                for j in self.selected_receivers
                for m in self.selected_days
                for sample in self.data['data'][i][j][m][1][:len(self.data['data'][i][j][m][1])//5]
            ]
            
            
        elif type == 'train' and train_test_split:

            cur_len = len(self.data['data'][0][0][0][1])
            
            self.data_ordered = [
                (sample, i)
                for i in self.devices
                for j in self.selected_receivers
                for m in self.selected_days
                for sample in self.data['data'][i][j][m][1][
                len(self.data['data'][i][j][m][1])//5:]
            ]
            
        else:
            self.data_ordered = [
                (sample, i)
                for i in self.devices
                for j in self.selected_receivers
                for m in self.selected_days
                for sample in self.data['data'][i][j][m][1]
            ]

    def __len__(self):
        return len(self.data_ordered)

    def fft_from_iq(self, sample):    
        data_i = sample[0]
        data_q = sample[1]
    
        f, t, spec = signal.stft(data_i + 1j*data_q,
                         window='boxcar',
                         nperseg=16,
                         noverlap=None,
                         nfft=128,
                         return_onesided=False,
                         padded=False,
                         boundary=None)
        
        spec = np.fft.fftshift(spec, axes=0)

        return np.abs(spec)[np.newaxis]

    def __getitem__(self, idx):
        sample, tx = self.data_ordered[idx]
        #tx = self.device_map[tx]
        
        sample = sample.T

        i_comp, q_comp = sample[0], sample[1]

        if self.cfo_compensate:
            # Blind per-burst bulk-CFO removal: estimate mean phase slope and de-rotate.
            x = i_comp.astype(np.float64) + 1j * q_comp.astype(np.float64)
            nidx = np.arange(x.shape[-1])
            cfo = np.mean(np.angle(x[1:] * np.conj(x[:-1]))) / (2 * np.pi)
            x = x * np.exp(-1j * 2 * np.pi * cfo * nidx)
            i_comp, q_comp = np.real(x), np.imag(x)
            sample = np.stack([i_comp, q_comp])

        # Cartesian -> polar (magnitude, phase in [0, 2pi)).
        magnitude = np.sqrt(i_comp ** 2 + q_comp ** 2)
        phase = np.arctan2(q_comp, i_comp)
        phase = (phase + 2 * np.pi) % (2 * np.pi)

        if self.polars:
            # Scale roughly into [0, 1] for both channels.
            sample = np.stack([magnitude / 1.5, phase / (2 * np.pi)])
        
        sample = sample.astype(np.float32)
        
        if self.transform_to_2d:
            wavelet = self.transform_to_2d['wavelet']
            scales = self.transform_to_2d['scales']
            resize_to = self.transform_to_2d['resize_to']

            sample = sample[0] + 1j * sample[1]
            coefs, _ = pywt.cwt(sample, np.arange(1, scales + 1), wavelet)
            sample = np.abs(coefs)[np.newaxis]
            sample = sample.astype(np.float32)
            #sample = np.stack([coefs.real, coefs.imag], axis=0)
            #sample = transforms.Resize(size=resize_to)(torch.tensor(sample, dtype=torch.float32))

        return (sample, tx, idx) if self.return_indices else (sample, tx)


class WiSig_Dataset_SingleDay(Dataset):
    """
    SingleDay compact WiSig subset (28 transmitters, 10 receivers, 1 day, 800
    signals/tx). Identical to WiSig_Dataset_ManyTx except that the number of
    devices is read from the pickle (28 here, not the hard-coded 150) and the
    ManyTx-specific 150-entry device_map permutation is replaced by an identity
    map (it only matters if you uncomment `tx = self.device_map[tx]` below).

    Source: S. Hanna, S. Karunaratne, and D. Cabric, "WiSig: A Large-Scale WiFi
    Signal Dataset for Receiver and Channel Agnostic RF Fingerprinting," IEEE
    Access, vol. 10, pp. 22808-22818, 2022, doi: 10.1109/ACCESS.2022.3154790.
    """
    def __init__(self, file: str, devices=None, days=None, selected_receivers=None,
                 return_indices=False, transform_to_2d=None, train_test_split=False,
                 type='train', k_fold_samples=0, polars=False, cfo_compensate=False):
        with open(file, 'rb') as f:
            self.data = pickle.load(f)
        self.transform_to_2d = transform_to_2d
        self.return_indices = return_indices
        self.polars = polars
        self.cfo_compensate = cfo_compensate
        self.selected_receivers = selected_receivers or list(range(len(self.data['rx_list'])))
        self.selected_days = days or list(range(len(self.data['capture_date_list'])))

        # Number of transmitters (classes) comes from the SingleDay pickle (= 28).
        num_devices = len(self.data['tx_list'])
        # ManyTx used a fixed 150-entry permutation here; SingleDay has its own
        # tx ordering, so use identity (the mapping line in __getitem__ is commented
        # out anyway, so labels are the raw device indices either way).
        self.device_map = list(range(num_devices))

        devices = devices or list(range(num_devices))
        self.devices = [i for i in range(num_devices) if i in devices]

        if type == 'validation' and train_test_split:
            cur_len = len(self.data['data'][0][0][0][0])

            self.data_ordered = [
                (sample, i)
                for i in self.devices
                for j in self.selected_receivers
                for m in self.selected_days
                for sample in self.data['data'][i][j][m][1][:len(self.data['data'][i][j][m][1]) // 5]
            ]

        elif type == 'train' and train_test_split:
            cur_len = len(self.data['data'][0][0][0][1])

            self.data_ordered = [
                (sample, i)
                for i in self.devices
                for j in self.selected_receivers
                for m in self.selected_days
                for sample in self.data['data'][i][j][m][1][
                    len(self.data['data'][i][j][m][1]) // 5:]
            ]

        else:
            self.data_ordered = [
                (sample, i)
                for i in self.devices
                for j in self.selected_receivers
                for m in self.selected_days
                for sample in self.data['data'][i][j][m][1]
            ]

    def __len__(self):
        return len(self.data_ordered)

    def fft_from_iq(self, sample):
        data_i = sample[0]
        data_q = sample[1]

        f, t, spec = signal.stft(data_i + 1j * data_q,
                                 window='boxcar',
                                 nperseg=16,
                                 noverlap=None,
                                 nfft=128,
                                 return_onesided=False,
                                 padded=False,
                                 boundary=None)

        spec = np.fft.fftshift(spec, axes=0)
        return np.abs(spec)[np.newaxis]

    def __getitem__(self, idx):
        sample, tx = self.data_ordered[idx]
        #tx = self.device_map[tx]

        sample = sample.T
        i_comp, q_comp = sample[0], sample[1]
        if self.cfo_compensate:
            # Blind per-burst bulk-CFO removal: estimate mean phase slope and de-rotate.
            x = i_comp.astype(np.float64) + 1j * q_comp.astype(np.float64)
            nidx = np.arange(x.shape[-1])
            cfo = np.mean(np.angle(x[1:] * np.conj(x[:-1]))) / (2 * np.pi)
            x = x * np.exp(-1j * 2 * np.pi * cfo * nidx)
            i_comp, q_comp = np.real(x), np.imag(x)
            sample = np.stack([i_comp, q_comp])
        # Cartesian -> polar (magnitude, phase in [0, 2pi)).
        magnitude = np.sqrt(i_comp ** 2 + q_comp ** 2)
        phase = np.arctan2(q_comp, i_comp)
        phase = (phase + 2 * np.pi) % (2 * np.pi)
        if self.polars:
            # Scale roughly into [0, 1] for both channels.
            sample = np.stack([magnitude / 1.5, phase / (2 * np.pi)])

        sample = sample.astype(np.float32)

        if self.transform_to_2d:
            wavelet = self.transform_to_2d['wavelet']
            scales = self.transform_to_2d['scales']
            resize_to = self.transform_to_2d['resize_to']
            sample = sample[0] + 1j * sample[1]
            coefs, _ = pywt.cwt(sample, np.arange(1, scales + 1), wavelet)
            sample = np.abs(coefs)[np.newaxis]
            sample = sample.astype(np.float32)
            #sample = np.stack([coefs.real, coefs.imag], axis=0)
            #sample = transforms.Resize(size=resize_to)(torch.tensor(sample, dtype=torch.float32))
        return (sample, tx, idx) if self.return_indices else (sample, tx)