# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 03:12:52 2024

@author: Dhrumit Patel
"""

from huggingface_hub import Repository

# Create a repository object, linked to your Hugging Face account
repo = Repository("HuggingFace_Clone_Final", clone_from="Dhrumit1314/BART_TextSummary")

# Add a files to the repository
# repo.git_add("config.json")
# repo.git_add("generation_config.json")
# repo.git_add("tf_model.h5")
repo.git_add("merges.txt")
repo.git_add("special_tokens_map.json")
repo.git_add("tokenizer_config.json")
repo.git_add("vocab.json")


# Commit the changes
repo.git_commit("Tokeizer files added Successfully !")

# Push the changes to the Hugging Face Model Hub
repo.git_push()






from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import TFBartForConditionalGeneration, BartTokenizer
import tensorflow as tf

model = TFBartForConditionalGeneration.from_pretrained("Dhrumit1314/BART_TextSummary")
