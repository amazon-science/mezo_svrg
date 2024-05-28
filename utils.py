import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data import random_split
import datasets
from torchvision.datasets import MNIST
from torchvision import transforms
import pickle
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from mezo_dataset import tokenize_multipart_input

matplotlib.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({'font.size': 12})

from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    GPT2Tokenizer
)

class MnistDataModule(pl.LightningDataModule):
    def __init__(self, bs=32, data_path='./'):
        super().__init__()
        self.data_path = data_path
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.batch_size = bs
        
    def prepare_data(self):
        MNIST(root=self.data_path, download=True) 

    def setup(self, stage=None):
        # stage is either 'fit', 'validate', 'test', or 'predict'
        # here note relevant
        mnist_all = MNIST( 
            root=self.data_path,
            train=True,
            transform=self.transform,  
            download=False
        ) 

        self.train, self.val = random_split(
            mnist_all, [55000, 5000], generator=torch.Generator().manual_seed(1)
        )

        self.test = MNIST( 
            root=self.data_path,
            train=False,
            transform=self.transform,  
            download=False
        ) 

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size,  shuffle=True, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size,  shuffle=True, num_workers=4)
    
    
class GLUEDataModule(pl.LightningDataModule):
    task_text_field_map = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "mrpc": ["sentence1", "sentence2"],
        "qqp": ["question1", "question2"],
        "stsb": ["sentence1", "sentence2"],
        "mnli": ["premise", "hypothesis"],
        "qnli": ["question", "sentence"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],
        "ax": ["premise", "hypothesis"],
    }

    glue_task_num_labels = {
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "stsb": 1,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
        "ax": 3,
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels"
    ]

    glue_task_label_mapping = {
        'sst2': {
            0: 'terrible',
            1: 'great'
        }
    }

    def __init__(
        self,
        model_name_or_path: str,
        task_name: str = "mrpc",
        max_seq_length: int = 512,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        sample_size: int = 128,
        validation_sample_size: int = 128,
        soft_prompt: bool=False,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.sample_size = sample_size
        self.validation_sample_size = validation_sample_size
        
        self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.glue_task_num_labels[task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        self.soft_prompt = soft_prompt
        if soft_prompt==True:
            self.loader_columns.append("mask_pos")


    def setup(self, stage: str):
        self.dataset = datasets.load_dataset("glue", self.task_name)

        for split in self.dataset.keys():
            if split=='test':
                continue
            if self.soft_prompt==True:
                self.dataset[split] = self.dataset[split].map(
                    self.convert_to_features_soft_prompt,
                    batched=False,
                    remove_columns=["label"],
                )
            else:
                self.dataset[split] = self.dataset[split].map(
                    self.convert_to_features,
                    batched=True,
                    remove_columns=["label"],
                )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]
        
        self.subset_indices = list(range(self.sample_size))
        self.subset_indices_val = list(range(self.validation_sample_size))
        self.subset_train_dataset = Subset(self.dataset["train"], self.subset_indices)
        if len(self.eval_splits) == 1:
            self.subset_val_dataset = Subset(self.dataset["validation"], self.subset_indices_val)
        else:
            self.subset_val_dataset = Subset(self.dataset["validation_matched"], self.subset_indices_val)

    def prepare_data(self):
        datasets.load_dataset("glue", self.task_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        #sampler = torch.utils.data.RandomSampler(self.dataset["train"], replacement=False, num_samples=self.sample_size)
        return DataLoader(self.subset_train_dataset, batch_size=self.train_batch_size, shuffle=True)
    
    def train_full_dataloader(self):
        #sampler = torch.utils.data.RandomSampler(self.dataset["train"], replacement=False, num_samples=self.sample_size)
        return DataLoader(self.subset_train_dataset, batch_size=self.sample_size, shuffle=True)

    def val_dataloader(self):
        #if len(self.eval_splits) == 1:
        #subset_indices = list(range(int(self.sample_size)))

        return DataLoader(self.subset_val_dataset, batch_size=self.eval_batch_size)
        #elif len(self.eval_splits) > 1:
        #    subset_indices = list(range(self.sample_size))
        #    return [print(x) for x in self.eval_splits] #DataLoader(Subset(self.dataset[x], subset_indices), batch_size=self.eval_batch_size)

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]
    
    def convert_to_features_soft_prompt(self, example_batch, indices=None):
        if self.task_name=='sst2':
            inputs = tokenize_multipart_input(
                input_text_list=[example_batch[self.text_fields[0]]],
                max_length=self.max_seq_length,
                tokenizer=self.tokenizer,
                task_name=self.task_name,
                prompt=True,
                template='*cls**sent_0*_It_was*mask*.*sep+*',
                label_word_list={0: ' terrible', 1: ' great'}
            )   
            inputs['labels'] = example_batch["label"] 
            return inputs
        elif self.task_name=='mnli':
            inputs = tokenize_multipart_input(
                input_text_list=[example_batch[self.text_fields[0]],example_batch[self.text_fields[1]]],
                max_length=256,
                tokenizer=self.tokenizer,
                task_name=self.task_name,
                prompt=True,
                template='*cls**sent-_0*?*mask*,*+sentl_1**sep+*',
                label_word_list={'contradiction':'No','entailment':'Yes','neutral':'Maybe'},
                first_sent_limit=240
            )   
            inputs['labels'] = example_batch["label"]
            return inputs

    def convert_to_features(self, example_batch, indices=None):

        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        if self.model_name_or_path == 'distilbert-base-cased' or self.model_name_or_path == 'roberta-large':
            features = self.tokenizer.batch_encode_plus(
                texts_or_text_pairs, max_length=self.max_seq_length, pad_to_max_length=True, truncation=True
            )
            # Rename label to labels to make it easier to pass to model forward
            features["labels"] = example_batch["label"]
        elif 'gpt2' in self.model_name_or_path or 'opt' in self.model_name_or_path:
            #self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            #features = self.tokenizer(texts_or_text_pairs, return_tensors='pt', truncation=True, padding=True)
            features = self.tokenizer.batch_encode_plus(
                texts_or_text_pairs, max_length=self.max_seq_length, pad_to_max_length=True, truncation=True
            )
            features["labels"] = example_batch["label"]

            # [torch.tensor() for i in range(1000)]
            # features["labels"] = torch.full((self.train_batch_size, self.max_seq_length), -100)
            # print('features tensor: ', features["labels"].shape)
            # print('label tensor: ', torch.tensor(example_batch["label"]).shape)
            # print('features input ids: ', torch.tensor(features["input_ids"]).shape)

            
            # features["labels"][: , -1] = torch.tensor(example_batch["label"])
            

        return features
    
def load_pickle(file_path, key):
    try:
        with open(file_path, 'rb') as file:
            # Load the data from the pickle file
            data = pickle.load(file)

            # Print the contents of the pickle file
            print("Contents of the pickle file:")
            if key == 'Tr_Loss':
                arr = windowed_mean(data[key], window_size=4)
                print(np.nanmin(arr))
            elif key == 'Val_Acc':
                arr = windowed_mean(data[key], window_size=4)
                print(np.max(arr))
            elif key == 'Query':
                print(np.sum(data[key]))
            else:
                print(data[key])

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")

    except Exception as e:
        print(f"Error: Unable to load the pickle file. {e}")
        
def convert_to_cpu_float(dictionary):
    """
    Convert PyTorch tensors in a dictionary from cuda:0 to floats on CPU.

    Parameters:
    - dictionary: Dictionary with keys and PyTorch tensors/lists of tensors on cuda:0.

    Returns:
    - Converted dictionary with tensors/lists as floats on CPU.
    """
    converted_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            converted_dict[key] = value.cpu().float()
        elif isinstance(value, list):
            converted_dict[key] = [item.cpu().numpy() for item in value]
        else:
            converted_dict[key] = value  # Non-tensor values remain unchanged

    return converted_dict


def average_across_batch(arr, epochs):
    arr_np = np.array(arr)
    arr_avg = np.mean(np.array(np.split(arr_np, epochs)), axis=1)
    #print(arr_avg.shape)
    return np.reshape(arr_avg, (1, arr_avg.shape[0]))

def moving_average(a, n=20):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_results(title, d, loss="Train Loss: ", lim_x=1000, lim_y=3):
    plt.figure()
    for k, val in d.items():

        v = np.array([val])
        means, mins, maxes = np.mean(v, axis=0), np.amin(v, axis=0), np.amax(v, axis=0)
        means = moving_average(means)
        plt.plot(range(1, len(means)+1), means, linewidth=2, linestyle='solid', markersize=12, label=k)
        #plt.fill_between(range(len(val['Tr_Loss'])), mins, maxes, alpha=0.5)
        
    plt.title(title)
    plt.yscale('log')
    plt.xlim(1, lim_x)
    plt.ylim(0.9, lim_y)
    plt.ylabel(loss)
    plt.xlabel('Step')
    plt.legend(bbox_to_anchor=(0.65, 1.0), loc="upper left")
    plt.grid()
    plt.tight_layout()
    plt.savefig(title + '.pdf')

def plot_results_time(title, d_y, d_x, loss="Train Loss: ", lim_x=1000, lim_y=3):
    plt.figure()
    for k, val in d_y.items():
        v = np.array([val])
        v_x = np.cumsum(np.array([d_x[k]]))*(1e-17)
        
        means, mins, maxes = np.mean(v, axis=0), np.amin(v, axis=0), np.amax(v, axis=0)
        means = moving_average(means)
        l = len(means)
        plt.plot(v_x[:l], means, linewidth=2, linestyle='solid', markersize=12, label=k)
        #plt.fill_between(range(len(val['Tr_Loss'])), mins, maxes, alpha=0.5)
        
    plt.title(title)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(0, lim_x)
    #plt.ylim(0.7, lim_y)
    plt.ylabel(loss)
    plt.xlabel('Time (s)')
    plt.legend(bbox_to_anchor=(0.0, 1), loc="upper left")
    plt.grid()
    plt.tight_layout()
    plt.savefig(title + '.pdf')
    
def plot_results_query(title, d_y, d_x, loss="Train Loss: ", lim_x=1000, lim_y=3):
    plt.figure()
    for k, val in d_y.items():
        v = np.array([val])
        v_x = np.cumsum(np.array([d_x[k]]))
        
        means, mins, maxes = np.mean(v, axis=0), np.amin(v, axis=0), np.amax(v, axis=0)
        means = moving_average(means)
        l = len(means)
        plt.plot(v_x[:l], means, linewidth=2, linestyle='solid', markersize=12, label=k)
        
    plt.title(title)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(1e3, lim_x)
    plt.ylim(0.7, lim_y)
    plt.ylabel(loss)
    plt.xlabel('Queries')
    plt.legend(bbox_to_anchor=(0.0, 1), loc="upper left")
    plt.grid()
    plt.tight_layout()
    plt.savefig(title + '.pdf')
        
def windowed_mean(arr, window_size):
    arr = np.array(arr)
    return np.mean(arr.reshape(-1, window_size), axis=1)
        
if __name__ == "__main__":
    #file_path = 'results_prelim_draft/mnli_ZO_SVRG_q2_lr0.001_bs64_samplesize1024_fullparamTrue.pickle'
    file_path = '/home/tgautamx/ZO_SmallScaleExp/gpt2/result_SST2_GPT2_FullParam/gpt2-xl_sst2_ZO_lr1e-06_bs64_samplesize512_fullparamTrue.pickle'
    key = 'Tr_Loss'
    load_pickle(file_path, key)
    