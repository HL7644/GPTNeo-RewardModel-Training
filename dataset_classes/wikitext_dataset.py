import torch

class WikiTextDataset(torch.utils.data.Dataset):
  def __init__(self, data, short):
    super(WikiTextDataset, self).__init__()
    self.short=short
    self.texts=self.get_text(data)
  
  def get_text(self, data):
    len_thresh=500
    texts=[]
    for dt in data:
      text=dt['text']
      if text!='' and len(text)>50:
        if (self.short and len(text)<len_thresh) or self.short==False:
          texts.append(text)
    return texts
  
  def __len__(self):
    return len(self.texts)
  
  def __getitem__(self, idx):
    return self.texts[idx]