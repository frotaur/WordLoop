from torch.utils.data import Dataset
import torch,os, h5py

class TokenTexth5(Dataset):
    """
        Dataset used to store tokenized text. Produces tuples of text, and the text shifted by one
        token, to be used as input and target for language modelling. Uses memory mapping, with hdf5.

        If we notice that creation of the data is SLOW, we may use batched calls like I did the the cellular automata, to be seen.
        Args:
        text_location : location of the tokenized text tensor
        attn_length : size of the attention window
        stride : by how many tokens to stride to get the next example. Default is half the attention length.
    """

    def __init__(self,h5_file :str, attn_length:int, stride:int=None, backwards=False):
        self.h5_file = h5_file
        self.attn_length = attn_length

        self.backwards = backwards

        
        if(stride is None):
            self.stride=self.attn_length//2
        else :
            self.stride = stride

        if(not os.path.isfile(self.h5_file)):
            raise ValueError(f'File/Folder {self.h5_file} not found')
        
        self.h5_file = h5py.File(self.h5_file, 'r')
        self.text_tensor = self.h5_file['tokens']


        self.num_tokens = len(self.text_tensor)
        self.length = (self.num_tokens-self.attn_length-1)//(self.stride) # -1 because we need to have a target for each input
    
        print(f'Dataset contains {self.num_tokens/1e6:.2f}M tokens, resulting in {self.length} examples.')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
            Returns a tuple of (input, target) tensors, each of shape (attn_length)

            For now, when backwards, we still give the examples in the 'forward' way, but
            we flip them. Maybe there is some reason why this is no bueno, but I don't think so.
        """
        if(self.backwards):
            true_idx = self.stride*(idx)+self.attn_length+1 # We still start from the front

            return torch.tensor(self.text_tensor[true_idx-self.attn_length:true_idx],dtype=torch.long).flip(dims=(0,)), \
                torch.tensor(self.text_tensor[true_idx-self.attn_length-1:true_idx-1],dtype=torch.long).flip(dims=(0,))
        else :
            true_idx = idx*self.stride
            return torch.tensor(self.text_tensor[true_idx:true_idx+self.attn_length],dtype=torch.long), \
            torch.tensor(self.text_tensor[true_idx+1:true_idx+self.attn_length+1],dtype=torch.long)



class TokenText(Dataset):
    """
        Dataset used to store tokenized text. Produces tuples of text, and the text shifted by one
        token, to be used as input and target for language modelling. For now, store the whole tokenized
        text on the CPU in one big tensor. Later, we might transition to something stored on disk, using
        h5py probably.

        Args:
        text_location : location of the tokenized text hdf5 dataset
        attn_length : size of the attention window
        stride : by how many tokens to stride to get the next example. Default is half the attention length.
        convert_on_the_fly : Whether to cast to Long on the fly, or once at the beginning (need to benchmark this)
    """

    def __init__(self,file_or_folder :str, attn_length:int, stride:int=None, backwards:bool=False,
                 convert_on_the_fly:bool = False):
        self.file_or_fold = file_or_folder
        self.attn_length = attn_length

        self.backwards = backwards
        self.convert_on_the_fly = convert_on_the_fly
        
        if(stride is None):
            self.stride=self.attn_length//2
        else :
            self.stride = stride
        
        self.text_tensor = torch.zeros((0,),dtype=torch.int32)
        if(os.path.isfile(self.file_or_fold)):
            self.text_tensor = torch.load(self.file_or_fold, map_location=torch.device('cpu'))[0,:]# (T) of int64 (maybe casting when getitem better, dunno)
        elif(os.path.isdir(self.file_or_fold)):
            for file in os.listdir(self.file_or_fold):
                if os.path.splitext(file)[1]=='.pt':
                    new_cat = torch.load(os.path.join(self.file_or_fold,file), map_location=torch.device('cpu'))[0,:]
                    self.text_tensor = torch.cat([self.text_tensor,new_cat],dim=0)
        else:
            raise ValueError(f'File/Folder {self.file_or_fold} not found')
        
        if(backwards):
            self.text_tensor = torch.flip(self.text_tensor,[0])

        self.num_tokens = self.text_tensor.shape[0]
        self.length = (self.num_tokens-self.attn_length-1)//(self.stride) # -1 because we need to have a target for each input
    
        print(f'Dataset contains {self.num_tokens/1e6:.2f}M tokens, resulting in {self.length} examples.')

        if(not convert_on_the_fly):
            self.text_tensor = self.text_tensor.to(torch.long)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
            Returns a tuple of (input, target) tensors, each of shape (attn_length)
        """
        if(self.convert_on_the_fly):
            return self.text_tensor[idx*self.stride:idx*self.stride+self.attn_length].to(torch.long), \
            self.text_tensor[idx*self.stride+1:idx*self.stride+self.attn_length+1].to(torch.long)
        else : 
            return self.text_tensor[idx*self.stride:idx*self.stride+self.attn_length], \
            self.text_tensor[idx*self.stride+1:idx*self.stride+self.attn_length+1]