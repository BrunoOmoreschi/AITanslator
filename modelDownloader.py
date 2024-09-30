#import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "facebook/m2m100_418M"
cache_dir = "D:\\AI\\iatxttranslator\\model"

# Download the model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir = cache_dir)

#Download finished on DELL 13/09/2024 15:38