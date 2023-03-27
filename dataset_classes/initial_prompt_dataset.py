import torch

#used for text generation
class InitialPromptDataset(torch.utils.data.Dataset):
  def __init__(self, hf_dset, short):
    super(InitialPromptDataset, self).__init__()
    #decompose hf_dset into state, action datasets
    self.short=short
    self.data=self.decompose_dset(hf_dset)
  
  def decompose_dset(self, hf_dset):
    len_thresh=500
    data=[]
    for hfd in hf_dset:
      chosen=hfd['chosen']
      if (self.short and len(chosen)<len_thresh) or self.short==False:
        initial_prompt=chosen.split('\n\n')[1]
        data.append(initial_prompt)
    return data
  
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    return self.data[idx]