NLP_Spam_Prediction

Flask application for detecting given message is spam or ham

bin
app.py : flask application file.
model.py : file for model creation and prediction.
preprocess.py : file for preprocess dataframe.

templates
home.html : html page for getting input message from user
result.hrml : html page for display prediction to user 
             Whether given message is spam or ham

cache 
nlp_model.pkl : nlp model save in pkl file
transform.pkl : CountVectorizer model save in pkl file

config
config.yml : configuration file for storing path

data
train
spam.csv : csv data file for trainning

Referance to www.youtube.com/channel/UCNU_lfiiWBdtULKOw6X0Dig
