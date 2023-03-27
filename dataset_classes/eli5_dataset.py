import torch
import numpy as np

class ELI5Dataset(torch.utils.data.Dataset):
  def __init__(self, dset, short):
    super(ELI5Dataset, self).__init__()
    self.short=short
    self.chosens, self.rejecteds=self.decompose_dset(dset)
  
  def decompose_dset(self, dset):
    len_thresh=500
    chosens_list=[]
    rejecteds_list=[]
    for dt in dset:
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
          if (self.short and len(rejected+title)<len_thresh) or self.short==False:
            rejecteds=rejecteds+rejected+"<SEP>"
        if len(rejecteds)>0:
          chosens_list.append(chosen)
          rejecteds_list.append(rejecteds[:-5]) #exclude last <SEP>
    return chosens_list, rejecteds_list

  def __len__(self):
    return len(self.chosens)
  
  def __getitem__(self, idx):
    return self.chosens[idx], self.rejecteds[idx]