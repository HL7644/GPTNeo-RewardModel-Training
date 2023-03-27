import torch

#used for classical RL training. + Instruct Fine Tuning w/o Davinci.
#also used for evaluating SFT models
class StateActionDataset(torch.utils.data.Dataset):
  def __init__(self, hf_dset, short):
    super(StateActionDataset, self).__init__()
    self.short=short
    self.chosens, self.states, self.actions=self.get_state_action_data(hf_dset)
  
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

  def get_state_action_data(self, hf_dset):
    len_thresh=500
    #take only chosen data for state action
    states=[]
    actions=[]
    chosens=[]
    for hfd in hf_dset:
      chosen=hfd['chosen']
      if (self.short and len(chosen)<len_thresh) or self.short==False:
        chosens.append(chosen)
        state, action=self.split_state_action_from_dialogue(chosen)
        states.append(state)
        actions.append(action)
    return chosens, states, actions
  
  def __len__(self):
    return len(self.states)
  
  def __getitem__(self, idx):
    return self.chosens[idx], self.states[idx], self.actions[idx]