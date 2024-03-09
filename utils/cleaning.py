import nltk
# nltk.download('wordnet')

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def drop_text_na(df):
    df = df.loc[(df['text_string'].isnull()==False)].copy()
    nan_value = float("NaN")
    df['text_string'].replace("", nan_value, inplace=True)
    df.dropna(subset = ['text_string'], inplace=True)
    return df

def clean_text_string(df):
     df['text_string'] = df.text_string.apply(lambda x: x.strip())
     df['text_string'].replace(to_replace ='\n', value = ' ', regex = True,
                               inplace=True)
     df['text_string'].replace(to_replace =' +', value = ' ', regex = True,
                               inplace=True)
     df['text_string'].replace(to_replace =r"[^a-zA-Z\d .,\/#!$%\^&\*;:{}=\-_`~()]+",
                            value = '', regex = True, inplace=True)
     return df

def init_cleaned_text_blocks(df):
    df['cleaned_text'] = df['text_string'].replace(to_replace = r"[^a-zA-Z\d ]+",
                                                   value = '', regex = True)
    return df

def init_cleaned_text_words(df):
    df['cleaned_text'] = df['text_string'].replace(to_replace = r"[^a-zA-Z ]+",
                                                   value = '', regex = True)
    return df

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text.lower())]

def block_lemmatize(df):
    df['text_token_lemmatized'] = df.cleaned_text.apply(lemmatize_text)
    return df    

def sent_create(word_list):
    return " ".join(word_list).strip()

def sent_cleaned_text(df):
    df['cleaned_text'] = df['text_token_lemmatized'].map(lambda x: sent_create(x))
    return df

def clean_block_df(df):
    df = df.pipe(drop_text_na).pipe(clean_text_string).pipe(drop_text_na).\
        pipe(init_cleaned_text_blocks).pipe(drop_text_na).\
            pipe(block_lemmatize).\
            pipe(sent_cleaned_text)         
    return df

def clean_words_df(df):
    df = df.pipe(drop_text_na).pipe(clean_text_string).pipe(drop_text_na).\
        pipe(init_cleaned_text_words).pipe(drop_text_na).\
            pipe(block_lemmatize).\
            pipe(sent_cleaned_text)
    return df


















