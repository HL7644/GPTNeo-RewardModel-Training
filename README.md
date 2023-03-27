# Reward Model Training using GPTNeo Architecture

**1. RLHF Dataset**

Running rlhf_dataset.ipynb downloads data from ELI5 and Anthropic HH-RLHF dataset. 

The data is preprocessed and saved as a pickle file

**2. Reward Model Training**

Running rm_training.ipynb trains reward model based on GPTNeo 125M.

Preferred response:
1. ELI5: answers with most votes
2. Anthropic HH-RLHF: given

**3. Supervised Fine Tuning**

Running sft.ipynb applys fine-tuning on pre-trained GPTNeo model based on question + preferred answer

Validation and Test Metrics are rewards given by Reward Model trained in 2.
