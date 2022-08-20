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

@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404

@app.route('/predict',methods=['POST'])
def predict():
    user_inpupt = request.form.values()
    new_review = [str(x) for x in user_inpupt]
    data = pd.DataFrame(new_review)
    data.columns = ['new_review']

    data['new_review'] = data['new_review'].apply(strip_tags)
#    data['new_review'] = [BeautifulSoup(text,"lxml").get_text() for text in data['new_review']]
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
    max_prob = predicted_prob[0,np.argmax(predicted_prob)]
    
    tag_indx = np.where(predicted_prob>=max_prob, 1, 0)
    max_prob = round(100 *max_prob, 2)

    # Inverse binarizer transform
    suggested_tag = loaded_w2v_mlb.inverse_transform(tag_indx) # .reshape(1,nbr_tags)
    suggested_tag = "'" + str(suggested_tag[0][0]) + "'"
    
    return render_template('base.html', suggested_tag=suggested_tag, predicted_prob=max_prob)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)

