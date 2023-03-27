import torch

#for SFT and generating trajectories for on-policy training
class ChosenDataset(torch.utils.data.Dataset):
  def __init__(self, hf_dset, short):
    super(ChosenDataset, self).__init__()
    self.short=short
    self.chosens=self.get_chosen(hf_dset)
  
  def get_chosen(self, hf_dset):
    len_thresh=500
    data=[]
    for hfd in hf_dset:
      chosen=hfd['chosen']
      if (self.short and len(chosen)<len_thresh) or self.short==False:
        #take only chosen data for SFT training
        data.append(chosen)
    return data
  
  def __len__(self):
    return len(self.chosens)
  
  def __getitem__(self, idx):
    return self.chosens[idx]