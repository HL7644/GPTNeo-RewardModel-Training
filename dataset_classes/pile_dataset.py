import torch

class PileDataset(torch.utils.data.Dataset):
  def __init__(self, data, short):
    super(PileDataset, self).__init__()
    self.short=short
    self.texts=self.get_text(data)
  
  def get_text(self, data):
    len_thresh=500
    texts=[]
    for dt in data:
      text_list=dt['texts']
      for text in text_list:
        if (self.short and len(text)<len_thresh) or self.short==False:
          texts.append(text)
    return texts
    
  def __len__(self):
    return len(self.texts)
  
  def __getitem__(self, idx):
    return self.texts[idx]