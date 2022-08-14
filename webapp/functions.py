# creating a function for data cleaning
import re
import spacy
nlp = spacy.load("en_core_web_sm")

import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from',  're', 'edu',  'not', 'would',
                   'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'try',
                   'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 
                   'think', 'see', 'rather', 'easy', 'lot', 
                   'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 
                   'right', 'line', 'even', 'also', 'may', 'take', 'come',
                   # perso
                   'like','work','have', 'code', 'file', 'use', 'one', 
                   'question','type','way','error','find', 'look'
])
special_tokens = ['c','r','c#','js']
from bs4 import BeautifulSoup

def strip_tags(html, whitelist=[]):
    """
    Strip all HTML tags except for a list of whitelisted tags.
    """
    soup = BeautifulSoup(html)

    for tag in soup.findAll(True):
        if tag.name not in whitelist:
            tag.append(' ')
            tag.replaceWithChildren()
    result = str(soup)

    # Clean up any repeated spaces and spaces like this: '<a>test </a> '
    result = re.sub(' +', ' ', result)
    result = re.sub(r' (<[^>]*> )', r'\1', result)
    result = re.sub('\n', ' ',result)
    return result.strip()

def pre_cleaner(x):
    # Case normalization
    x = x.lower()
    # Remove unicode characters
    x = x.encode("ascii", "ignore").decode()
    # Remove links
    x = re.sub(r'http*\S+', '', x)
    # Remove numbers
    x = re.sub(r'\w*\d+\w*', '', x)
    # Remove extra spaces
    x = re.sub(r'\s+', ' ', x)
    # Remove url
    x = re.sub(r'https?:\S+|http?:\S', ' ', x)
    # Replace special character
    x = x.replace('-', ' ').replace('/', ' ').replace(':',' ').replace("'", ' ')\
         .replace('=', ' ').replace('..',' ').replace('...',' ').replace(',',' ')\
         .replace('(',' ').replace(')',' ').replace('*',' ')\
         .replace('_',' ').replace('  ', ' ').replace('-',' ')
    # Remove ponctuation but not # (for C# for example)
#     x = re.sub('[^\\w\\s#|+]', '', x)
    return x
    
def tokenizer_fct(sentence):
    sentence = sentence.replace('.', ' ')
#     sentence = sentence.replace('-',' ').replace('.',' ').replace('/',' ').replace(':',' ')\
#                            .replace('=',' ').replace('..',' ').replace('...',' ')\
#                            .replace('(',' ').replace(')',' ').replace('*',' ')\
#                            .replace('_',' ').replace('  ',' ')
#     sentence = re.sub('[^\\w\\s#|+]', '', sentence)
    word_tokens = word_tokenize(sentence)
    return word_tokens

def stop_word_filter_fct(list_words, stop_words=stop_words):
    filtered_w = [w for w in list_words if not w in stop_words or (w in special_tokens)]
    filtered_w2 = [w for w in filtered_w if (len(w) > 2) or (w in special_tokens)]
    return filtered_w2

def transform_bow_fct(desc_text):
#     desc_text = desc_text.replace('.', ' ')
    word_tokens = tokenizer_fct(desc_text)
    word_tokens = keep_hashtag_token(word_tokens)
    sw = stop_word_filter_fct(word_tokens)
    transf_desc_text = ' '.join(sw)
    return transf_desc_text

def lemmatizer_spacy(sentence, stop_words=stop_words):
    doc = nlp(sentence)
    lemmas = []
    for token in doc:
#         if not token.is_punct:
        temp_lm = token.lemma_
        if temp_lm not in stop_words:
            lemmas.append(temp_lm)
        final_lemmas = keep_hashtag_token(lemmas)
    return " ".join(final_lemmas) # str(lemmas) # 

# Pour garder "C#"
def keep_hashtag_token(tokens):
    i_offset = 0
    for i, t in enumerate(tokens):
        i -= i_offset
        if t == '#' and i > 0:
            left = tokens[:i-1]
            joined = [tokens[i - 1] + t]
            right = tokens[i + 1:]
            tokens = left + joined + right
            i_offset += 1
    return tokens