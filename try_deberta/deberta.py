#TODO
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
#%%
classifier("We are very happy to show you the 🤗 Transformers library.")

#%%
results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

#%%
import torch
from transformers import pipeline

speech_recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

#%%
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)



#%%
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
classifier("Nous sommes très heureux de vous présenter la bibliothèque 🤗 Transformers.")

#%%
from transformers import AutoTokenizer

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)

#%%
encoding = tokenizer("We are very happy to show you the 🤗 Transformers library.")
print(encoding)

#%%
tf_batch = tokenizer(
    ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="tf",
)
print(tf_batch)


#%%
from transformers import TFAutoModelForSequenceClassification

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tf_model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
tf_outputs = tf_model(tf_batch)
print(tf_outputs)
print(type(tf_outputs))
logits_shape = tf_outputs.logits.shape
print("Shape of logits:", logits_shape)

#%%
import tensorflow as tf

tf_predictions = tf.nn.softmax(tf_outputs.logits, axis=-1)
tf_predictions


#%%
import transformers
from transformers import AutoModel,AutoTokenizer

bert_name="bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(bert_name)
BERT = AutoModel.from_pretrained(bert_name)

e=tokenizer.encode('I am hoping for the best', add_special_tokens=True)

q=BERT(torch.tensor([e]))
print (q[1].shape)
print(q[1])


#%%
#from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import TFAutoModel,AutoTokenizer
#import tensorflow as tf

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge")
model = TFAutoModel.from_pretrained("microsoft/deberta-v2-xlarge")

# Tokenize and process input
inputs = tokenizer(
    [
        "Hello, my dog is cute",
        "Hello, my dog is cute 2",
        "Hello, my dog is cute 2 3",
     ],
                   padding=True,
                   truncation=True,
                   max_length=512,
                   add_special_tokens=True,
                   return_tensors="tf")
outputs = model(**inputs)
#print(outputs)
# Extract the 'pooled_output'
# The first element of each batch's last_hidden_state is the poolled output

print(outputs.last_hidden_state[:, 0:1, :])
