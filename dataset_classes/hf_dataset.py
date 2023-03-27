import torch

#used for InstructGPT style reward model training
class HFDataset(torch.utils.data.Dataset):
  def __init__(self, hf_dset, short):
    super(HFDataset, self).__init__()
    self.short=short
    self.chosens, self.rejecteds=self.decompose_dset(hf_dset)
  
  def decompose_dset(self, hf_dset):
    len_thresh=500
    chosens=[]
    rejecteds=[]
    for hfd in hf_dset:
      chosen=hfd['chosen']
      if (self.short and len(chosen)<len_thresh) or self.short==False:
        rejected=hfd['rejected']
        chosens.append(chosen)
        rejecteds.append(rejected)
    return chosens, rejecteds

  def __len__(self):
    return len(self.chosens)
  
  def __getitem__(self, idx):
    return self.chosens[idx], self.rejecteds[idx]