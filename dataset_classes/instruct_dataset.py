import torch

class InstructDataset(torch.utils.data.Dataset):
  def __init__(self, hf_state_action_dset, eli5_dset, short):
    super(InstructDataset, self).__init__()
    self.short=short
    self.chosens, self.states, self.actions=self.decompose_dset(hf_state_action_dset, eli5_dset)
  
  def split_state_action_from_dialogue(self, text):
    #type=>  Human: ..., Assistant: ...
    #corrects typos.
    text=text.replace("Humans:","Human:")
    text=text.replace("human:","Human:")
    text=text.replace("humans:","Human:")
    assistant_splitted=text.split("Assistant: ")
    action="Assistant: "+assistant_splitted[-1]
    state=text.replace(action,"")
    return state, action
  
  def decompose_dset(self, hf_state_action_dset, eli5_dset):
    chosens_list=[]
    states_list=[]
    actions_list=[]
    #add hf state action dset
    chosens_list.extend(hf_state_action_dset.chosens)
    states_list.extend(hf_state_action_dset.states)
    actions_list.extend(hf_state_action_dset.actions)
    #handle ELI5 data
    for eli5_data in eli5_dset:
      #eli data are formatted into chosens, rejecteds
      #only use chosen for SFT
      chosen, _=eli5_data
      state, action=self.split_state_action_from_dialogue(chosen)
      chosens_list.append(chosen)
      states_list.append(state)
      actions_list.append(action)
    return chosens_list, states_list, actions_list
  
  def __len__(self):
    return len(self.chosens)
  
  def __getitem__(self, idx):
    return self.chosens[idx], self.states[idx], self.actions[idx]