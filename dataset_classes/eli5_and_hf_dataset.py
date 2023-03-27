import torch
import numpy as np

class ELI5andHFDataset(torch.utils.data.Dataset):
  def __init__(self, eli5_dset, hf_dset, short):
    super(ELI5andHFDataset, self).__init__()
    self.short=short
    self.chosens, self.rejecteds=self.decompose_dset(eli5_dset, hf_dset)
  
  def decompose_dset(self, eli5_dset, hf_dset):
    len_thresh=500
    chosens_list=[]
    rejecteds_list=[]
    #handle eli5 dataset
    for dt in eli5_dset:
      title=dt['title']
      answers=dt['answers']['text']
      scores=dt['answers']['score']
      selftext=dt['selftext']
      argmax_idx=np.argmax(scores)
      if self.short==False and selftext!='':
        chosen="Human: "+title+'\n'+selftext+"\n\n"+"Assistant: "+answers[argmax_idx]
      else:
        chosen="Human: "+title+"\n\n"+"Assistant: "+answers[argmax_idx]
      answers.pop(argmax_idx)
      rejecteds=""
      if (self.short and len(chosen)<len_thresh) or self.short==False:
        for rej in answers:
          rejected="Human: "+title+"\n\n"+"Assistant: "+rej
          #combine answers into a single string with <SEP> token to decompose afterwards.
          if (self.short and len(rejected+title)<len_thresh) or self.short==False:
            rejecteds=rejecteds+rejected+"<SEP>"
        if len(rejecteds)>0:
          chosens_list.append(chosen)
          rejecteds_list.append(rejecteds[:-5]) #remove last <SEP>
    #handle hf dataset
    for hfd in hf_dset:
      chosen=hfd['chosen']
      if (self.short and len(chosen)<len_thresh) or self.short==False:
        rejected=hfd['rejected']
        chosens_list.append(chosen)
        rejecteds_list.append(rejected)
    return chosens_list, rejecteds_list

  def __len__(self):
    return len(self.chosens)
  
  def __getitem__(self, idx):
    return self.chosens[idx], self.rejecteds[idx]

