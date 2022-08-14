# Library imports
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pickle
from functions import *

# load from models
# binarizer
filename_mlb = 'models/w2v_mlb.pickle'
loaded_w2v_mlb = pickle.load(open(filename_mlb, 'rb'))
# tokenizer
filename_tokenizer = 'models/w2v_tokenizer.pickle'
loaded_w2v_tokenizer = pickle.load(open(filename_tokenizer, 'rb'))
# model
filename_model = 'models/best_w2v_maxlen64.hdf5'
loaded_w2v_model = load_model(filename_model)


# Create the app object
app = Flask(__name__)
# Define predict function
@app.route('/')
def home():
    return render_template('base.html')

@app.route('/predict',methods=['POST'])
def predict():
    user_inpupt = request.form.values()
    new_review = [str(x) for x in user_inpupt]
    data = pd.DataFrame(new_review)
    data.columns = ['new_review']

    data['new_review'] = data['new_review'].apply(strip_tags)
    data['new_review'] = [BeautifulSoup(text,"lxml").get_text() for text in data['new_review']]
    data['review_pc'] = data['new_review'].apply(lambda x : pre_cleaner(x))
    data['review_pc'] = data['review_pc'].apply(lambda x : transform_bow_fct(x))
    data['review_lm'] = data['review_pc'].apply(lambda x : lemmatizer_spacy(x))

    maxSeq_len = loaded_w2v_model.input_shape[1]

    eval_sequences = loaded_w2v_tokenizer.texts_to_sequences(data.review_lm) 
    eval_padded = pad_sequences(
        eval_sequences, 
        maxlen=maxSeq_len, 
        truncating="post", 
        padding="post"
    )

    predicted_prob = loaded_w2v_model.predict(eval_padded)

    tresh = 0.3
    predicted_prob[predicted_prob >= tresh] = 1
    predicted_prob[predicted_prob < tresh] = 0

    # Inverse binarizer transform
    y_test_pred_inversed = loaded_w2v_mlb.inverse_transform(predicted_prob)
    
    return render_template('base.html', prediction_text=y_test_pred_inversed[0])

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)

