import torch

class OffPolicyDataset(torch.utils.data.Dataset):
  def __init__(self, hf_ep_steps_dset, eli5_dset, short):
    super(OffPolicyDataset, self).__init__()
    self.short=short
    self.ep_steps=self.decompose_dset(hf_ep_steps_dset, eli5_dset)
  
  def get_ep_steps_from_dialogue(self, text):
    #type=>  Human: ..., Assistant: ...
    #corrects typos.
    text=text.replace("Humans:","Human:")
    text=text.replace("human:","Human:")
    text=text.replace("humans:","Human:")
    human_splitted=text.split("Human: ")
    human_parts=[]
    assistant_parts=[]
    for hs in human_splitted:
      if "Assistant: " in hs:
        assistant_splitted=hs.split("Assistant: ")
        human_part=assistant_splitted[0]
        assistant_part_list=assistant_splitted[1:]
        assistant_part=""
        for ap in assistant_part_list:
          if ap!='\n\n' or ap!='\n' or ap!='':
            assistant_part+=ap
        human_part="Human: "+human_part
        human_parts.append(human_part)
        assistant_part="Assistant: "+assistant_part
        assistant_parts.append(assistant_part)
    #pack human part and assistant part as ep_steps of format (state, action, state_f)
    len_parts=len(human_parts)
    ep_steps=[]
    state=""
    for idx in range(len_parts):
      state=state+human_parts[idx]
      action=assistant_parts[idx]
      if idx==len_parts-1:
        state_f=None
        termin_signal=True
      else:
        state_f=state+action+human_parts[idx+1]
        termin_signal=False
      ep_step={
          'state': state,
          'action': action,
          'state+action': state+action, #used for action value computation
          'state_f': state_f,
          'termin_signal': termin_signal
      }
      state=state+action
      ep_steps.append(ep_step)
    return ep_steps
  
  def decompose_dset(self, hf_ep_steps_dset, eli5_dset):
    ep_steps=[]
    ep_steps.extend(hf_ep_steps_dset.ep_steps)
    for eli5_data in eli5_dset:
      chosen, _=eli5_data
      eps=self.get_ep_steps_from_dialogue(chosen)
      ep_steps.extend(eps)
    return ep_steps
  
  def __len__(self):
    return len(self.ep_steps)
  
  def __getitem__(self, idx):
    return self.ep_steps[idx]