import streamlit as st
import pandas as pd
import string
import re
import numpy as np
import matplotlib.pyplot as plt
import nltk
import pickle
from plotly import graph_objs as go
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')

abbreviations = {
    "$": " dollar ",
    "€": " euro ",
    "4ao": "for adults only",
    "a.m": "before midday",
    "a3": "anytime anywhere anyplace",
    "aamof": "as a matter of fact",
    "acct": "account",
    "adih": "another day in hell",
    "afaic": "as far as i am concerned",
    "afaict": "as far as i can tell",
    "afaik": "as far as i know",
    "afair": "as far as i remember",
    "afk": "away from keyboard",
    "app": "application",
    "approx": "approximately",
    "apps": "applications",
    "asap": "as soon as possible",
    "asl": "age, sex, location",
    "atk": "at the keyboard",
    "ave.": "avenue",
    "aymm": "are you my mother",
    "ayor": "at your own risk",
    "b&b": "bed and breakfast",
    "b+b": "bed and breakfast",
    "b.c": "before christ",
    "b2b": "business to business",
    "b2c": "business to customer",
    "b4": "before",
    "b4n": "bye for now",
    "b@u": "back at you",
    "bae": "before anyone else",
    "bak": "back at keyboard",
    "bbbg": "bye bye be good",
    "bbc": "british broadcasting corporation",
    "bbias": "be back in a second",
    "bbl": "be back later",
    "bbs": "be back soon",
    "be4": "before",
    "bfn": "bye for now",
    "blvd": "boulevard",
    "bout": "about",
    "brb": "be right back",
    "bros": "brothers",
    "brt": "be right there",
    "bsaaw": "big smile and a wink",
    "btw": "by the way",
    "bwl": "bursting with laughter",
    "c/o": "care of",
    "cet": "central european time",
    "cf": "compare",
    "cia": "central intelligence agency",
    "csl": "can not stop laughing",
    "cu": "see you",
    "cul8r": "see you later",
    "cv": "curriculum vitae",
    "cwot": "complete waste of time",
    "cya": "see you",
    "cyt": "see you tomorrow",
    "dae": "does anyone else",
    "dbmib": "do not bother me i am busy",
    "diy": "do it yourself",
    "dm": "direct message",
    "dwh": "during work hours",
    "e123": "easy as one two three",
    "eet": "eastern european time",
    "eg": "example",
    "embm": "early morning business meeting",
    "encl": "enclosed",
    "encl.": "enclosed",
    "etc": "and so on",
    "faq": "frequently asked questions",
    "fawc": "for anyone who cares",
    "fb": "facebook",
    "fc": "fingers crossed",
    "fig": "figure",
    "fimh": "forever in my heart",
    "ft.": "feet",
    "ft": "featuring",
    "ftl": "for the loss",
    "ftw": "for the win",
    "fwiw": "for what it is worth",
    "fyi": "for your information",
    "g9": "genius",
    "gahoy": "get a hold of yourself",
    "gal": "get a life",
    "gcse": "general certificate of secondary education",
    "gfn": "gone for now",
    "gg": "good game",
    "gl": "good luck",
    "glhf": "good luck have fun",
    "gmt": "greenwich mean time",
    "gmta": "great minds think alike",
    "gn": "good night",
    "g.o.a.t": "greatest of all time",
    "goat": "greatest of all time",
    "goi": "get over it",
    "gps": "global positioning system",
    "gr8": "great",
    "gratz": "congratulations",
    "gyal": "girl",
    "h&c": "hot and cold",
    "hp": "horsepower",
    "hr": "hour",
    "hrh": "his royal highness",
    "ht": "height",
    "ibrb": "i will be right back",
    "ic": "i see",
    "icq": "i seek you",
    "icymi": "in case you missed it",
    "idc": "i do not care",
    "idgadf": "i do not give a damn fuck",
    "idgaf": "i do not give a fuck",
    "idk": "i do not know",
    "ie": "that is",
    "i.e": "that is",
    "ifyp": "i feel your pain",
    "IG": "instagram",
    "iirc": "if i remember correctly",
    "ilu": "i love you",
    "ily": "i love you",
    "imho": "in my humble opinion",
    "imo": "in my opinion",
    "imu": "i miss you",
    "iow": "in other words",
    "irl": "in real life",
    "j4f": "just for fun",
    "jic": "just in case",
    "jk": "just kidding",
    "jsyk": "just so you know",
    "l8r": "later",
    "lb": "pound",
    "lbs": "pounds",
    "ldr": "long distance relationship",
    "lmao": "laugh my ass off",
    "lmfao": "laugh my fucking ass off",
    "lol": "laughing out loud",
    "ltd": "limited",
    "ltns": "long time no see",
    "m8": "mate",
    "mf": "motherfucker",
    "mfs": "motherfuckers",
    "mfw": "my face when",
    "mofo": "motherfucker",
    "mph": "miles per hour",
    "mr": "mister",
    "mrw": "my reaction when",
    "ms": "miss",
    "mte": "my thoughts exactly",
    "nagi": "not a good idea",
    "nbc": "national broadcasting company",
    "nbd": "not big deal",
    "nfs": "not for sale",
    "ngl": "not going to lie",
    "nhs": "national health service",
    "nrn": "no reply necessary",
    "nsfl": "not safe for life",
    "nsfw": "not safe for work",
    "nth": "nice to have",
    "nvr": "never",
    "nyc": "new york city",
    "oc": "original content",
    "og": "original",
    "ohp": "overhead projector",
    "oic": "oh i see",
    "omdb": "over my dead body",
    "omg": "oh my god",
    "omw": "on my way",
    "p.a": "per annum",
    "p.m": "after midday",
    "pm": "prime minister",
    "poc": "people of color",
    "pov": "point of view",
    "pp": "pages",
    "ppl": "people",
    "prw": "parents are watching",
    "ps": "postscript",
    "pt": "point",
    "ptb": "please text back",
    "pto": "please turn over",
    "qpsa": "what happens",  # "que pasa",
    "ratchet": "rude",
    "rbtl": "read between the lines",
    "rlrt": "real life retweet",
    "rofl": "rolling on the floor laughing",
    "roflol": "rolling on the floor laughing out loud",
    "rotflmao": "rolling on the floor laughing my ass off",
    "rt": "retweet",
    "ruok": "are you ok",
    "sfw": "safe for work",
    "sk8": "skate",
    "smh": "shake my head",
    "sq": "square",
    "srsly": "seriously",
    "ssdd": "same stuff different day",
    "tbh": "to be honest",
    "tbs": "tablespooful",
    "tbsp": "tablespooful",
    "tfw": "that feeling when",
    "thks": "thank you",
    "tho": "though",
    "thx": "thank you",
    "tia": "thanks in advance",
    "til": "today i learned",
    "tl;dr": "too long i did not read",
    "tldr": "too long i did not read",
    "tmb": "tweet me back",
    "tntl": "trying not to laugh",
    "ttyl": "talk to you later",
    "u": "you",
    "u2": "you too",
    "u4e": "yours for ever",
    "utc": "coordinated universal time",
    "w/": "with",
    "w/o": "without",
    "w8": "wait",
    "wassup": "what is up",
    "wb": "welcome back",
    "wtf": "what the fuck",
    "wtg": "way to go",
    "wtpa": "where the party at",
    "wuf": "where are you from",
    "wuzup": "what is up",
    "wywh": "wish you were here",
    "yd": "yard",
    "ygtr": "you got that right",
    "ynk": "you never know",
    "zzz": "sleeping bored and tired"
}

# Remove all english stopwords
def remove_stopwords(text):
    text = ' '.join([word for word in text.split() if word not in stopwords.words("english")])
    return text

# Remove all punctuations
def remove_all_punct(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)

# Remove all URLs, replace by URL
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'URL', str(text))

# Remove HTML beacon
def remove_HTML(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

# Remove non printable characters
def remove_not_ASCII(text):
    text = ''.join([word for word in text if word in string.printable])
    return text

# Change an abbreviation by its true meaning
def word_abbrev(word):
    return abbreviations[word.lower()] if word.lower() in abbreviations.keys() else word

# Replace all abbreviations
def replace_abbrev(text):
    string = ""
    for word in text.split():
        string += word_abbrev(word) + " "
    return string

# Remove @ and mention, replace by USER
def remove_mention(text):
    at = re.compile(r'@\S+')
    return at.sub(r'USER', text)

# Remove numbers, replace it by NUMBER
def remove_number(text):
    num = re.compile(r'[-+]?[.\d]*[\d]+[:,.\d]*')
    return num.sub(r'NUMBER', text)

# Remove emoji
def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

# lemmatize text (convert to root form of text)
def lemmatization(text):
    lm = WordNetLemmatizer()
    text = ' '.join([lm.lemmatize(word, pos='v') for word in text.split()])
    return text

def clean_text(text):
    # lowercase text and remove punctuation
    text = text.lower()
    text = remove_all_punct(text)

    # Remove non text (url, html tag, non ASCII characters)
    text = remove_URL(text)
    text = remove_HTML(text)
    text = remove_not_ASCII(text)

    # replace abbreviations, remove mention, remove emoji, remove number
    text = replace_abbrev(text)
    text = remove_mention(text)
    text = remove_emoji(text)
    text = remove_number(text)

    # Remove stopwords and lemmatize text
    text = remove_stopwords(text)
    text = lemmatization(text)
    return text

# n-grams
def get_ngrams(text, ngram_from=2, ngram_to=2, n=None):
    vec = CountVectorizer(ngram_range=(ngram_from, ngram_to)).fit(text)
    bag_of_words = vec.transform(text)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, i]) for word, i in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

    return words_freq[:n]

# generate word cloud
def wordcloud_draw(data, color='black'):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                             if 'http' not in word
                             and not word.startswith('@')
                             and not word.startswith('#')
                             and word != 'RT'])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color=color,
                          width=2000,
                          height=1500,
                          min_font_size=20,
                          max_font_size=200
                          ).generate(cleaned_word)
    # Display the generated image:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(wordcloud)
    plt.axis("off")
    st.pyplot(fig)

# prediction
def predict_sentiment_batch(review):
    label_list = []
    for input_text in review:
        cleaned_text = clean_text(input_text)
        text_tv = vectorizer.transform([cleaned_text])
        prediction = model.predict(text_tv)
        label_list.append(prediction)
    output = pd.DataFrame(data=label_list, columns=['label'])
    return output

if 'stage' not in st.session_state:
    st.session_state.stage = 0

def set_stage(stage):
    st.session_state.stage = stage

# Main code
st.set_page_config(layout='wide')
with st.sidebar:
    st.write('This page can analyze the sentiment of multiple reviews.')
    st.image('Image 2.png')
    st.write('1. Upload a file in the required format and let the analyzer do the work!')
    st.write('2. A table with the review text and their sentiment labels will be displayed.')
    st.write('  - Click Download Output button if you want to download the output table.')
    st.write('3. Choose the type of visualization')
    st.write('  - Click Bar Chart button to view the distribution of the sentiment labels.')
    st.write('  - Click Wordcloud button to view the words contain in positive and negative sentiment sentence.')

st.header("Batch Review Prediction")
st.write("Upload reviews as .csv file which contains the review's column with **column name = text** See example below:")
example = pd.read_csv("example.csv")
st.write(example.head())
upload_file = st.file_uploader("Note: If the uploaded file is large, it may take up to few minutes.", type=["csv"])

model_path = 'finalized_model.pkl'
vectorizer_path = 'vectorizer.pkl'
model = pickle.load(open(model_path, 'rb'))
vectorizer = pickle.load(open(vectorizer_path, 'rb'))

submitted = st.button('Analyze', on_click=set_stage, args=(1,))
if st.session_state.stage > 0:
    if upload_file is not None:
        df = pd.read_csv(upload_file, encoding='latin-1')
        input_list = df['text'].tolist()
        predict_output = pd.DataFrame(predict_sentiment_batch(input_list))
        result_df = df.assign(label=predict_output)
        st.subheader('Result')
        st.write(result_df)
        @st.cache
        def convert_df(data):
            # Cache the conversion to prevent computation on every rerun
            return data.to_csv().encode('utf-8')
        csv = convert_df(result_df)
        st.download_button(label="Download Output Data", data=csv, file_name='output.csv', mime='text/csv')

        st.subheader('Visualization')
        viz_option = st.radio('Choose plot', ('Bar Chart', 'Word Cloud', 'N-grams'), horizontal=True)
        plot_button = st.button('Plot', on_click=set_stage, args=(2,))
        if st.session_state.stage > 1:
            if viz_option == 'Word Cloud':
                sentiment_choice = st.selectbox('Select sentiment', ('Positive', 'Negative', 'Neutral'))
                if sentiment_choice == "Positive":
                    review_pos = result_df[result_df['label'] == 'positive']
                    review_pos = review_pos['text']
                    st.subheader("Words contain in positive reviews")
                    wordcloud_draw(review_pos, 'white')
                elif sentiment_choice == "Negative":
                    review_neg = result_df[result_df['label'] == 'negative']
                    review_neg = review_neg['text']
                    st.subheader("Words contain in negative reviews")
                    wordcloud_draw(review_neg)
                else:
                    review_neu = result_df[result_df['label'] == 'neutral']
                    review_neu = review_neu['text']
                    st.subheader("Words contain in neutral reviews")
                    wordcloud_draw(review_neu)

            elif viz_option == 'Bar Chart':
                count = result_df.groupby('label').count()['text'].reset_index().sort_values(by='text', ascending=False)
                fig = go.Figure(go.Bar(x=count.label, y=count.text, text=count.text))
                fig.show()
                st.subheader("Bar Chart of Sentiment Distribution")
                st.plotly_chart(fig)

            elif viz_option == 'N-grams':
                ngram_option = st.selectbox("Select n-grams (Number of word sequence)", ("Unigram", "Bigram", "Trigram"))
                review_pos = result_df[result_df['label'] == 'positive']
                review_neu = result_df[result_df['label'] == 'neutral']
                review_neg = result_df[result_df['label'] == 'negative']
                if ngram_option == "Unigram":
                    # positive unigram
                    unigrams_pos_df = pd.DataFrame(get_ngrams(review_pos['text'], ngram_from=1, ngram_to=1, n=15))
                    unigrams_pos_df.columns = ["Unigram", "Frequency"]
                    unigrams_pos_df = unigrams_pos_df.head(10).sort_values(by='Frequency', ascending=True)
                    fig, ax = plt.subplots()
                    ax.barh("Unigram", "Frequency", color='green', height=0.4, data=unigrams_pos_df)
                    st.write("Top 10 words in positive reviews - UNIGRAM ANALYSIS")
                    st.pyplot(fig)

                    # negative unigram
                    unigrams_neg_df = pd.DataFrame(get_ngrams(review_neg['text'], ngram_from=1, ngram_to=1, n=15))
                    unigrams_neg_df.columns = ["Unigram", "Frequency"]
                    unigrams_neg_df = unigrams_neg_df.head(10).sort_values(by='Frequency', ascending=True)
                    fig, ax = plt.subplots()
                    plt.barh("Unigram", "Frequency", color='red', height=0.4, data=unigrams_neg_df)
                    st.write("Top 10 words in negative reviews - UNIGRAM ANALYSIS")
                    st.pyplot(fig)

                elif ngram_option == "Bigram":
                    # positive bigram
                    bigrams_pos_df = pd.DataFrame(get_ngrams(review_pos['text'], ngram_from=2, ngram_to=2, n=15))
                    bigrams_pos_df.columns = ["Bigram", "Frequency"]
                    bigrams_pos_df = bigrams_pos_df.head(10).sort_values(by='Frequency', ascending=True)
                    fig, ax = plt.subplots()
                    ax.barh("Bigram", "Frequency", color='green', height=0.4, data=bigrams_pos_df)
                    st.write("Top 10 words in positive reviews - BIGRAM ANALYSIS")
                    st.pyplot(fig)

                    # negative bigram
                    bigrams_neg_df = pd.DataFrame(get_ngrams(review_neg['text'], ngram_from=2, ngram_to=2, n=15))
                    bigrams_neg_df.columns = ["Bigram", "Frequency"]
                    bigrams_neg_df = bigrams_neg_df.head(10).sort_values(by='Frequency', ascending=True)
                    fig, ax = plt.subplots()
                    ax.barh("Bigram", "Frequency", color='red', height=0.4, data=bigrams_neg_df)
                    st.write("Top 10 words in negative reviews - BIGRAM ANALYSIS")
                    st.pyplot(fig)

                elif ngram_option == "Trigram":
                    # positive trigram
                    trigrams_pos_df = pd.DataFrame(get_ngrams(review_pos['text'], ngram_from=3, ngram_to=3, n=15))
                    trigrams_pos_df.columns = ["Trigram", "Frequency"]
                    trigrams_pos_df = trigrams_pos_df.head(10).sort_values(by='Frequency', ascending=True)
                    fig, ax = plt.subplots()
                    ax.barh("Trigram", "Frequency", color='green', height=0.4, data=trigrams_pos_df)
                    st.write("Top 10 words in positive reviews - TRIGRAM ANALYSIS")
                    st.pyplot(fig)

                    # negative trigram
                    trigrams_neg_df = pd.DataFrame(get_ngrams(review_neg['text'], ngram_from=3, ngram_to=3, n=15))
                    trigrams_neg_df.columns = ["Trigram", "Frequency"]
                    trigrams_neg_df = trigrams_neg_df.head(10).sort_values(by='Frequency', ascending=True)
                    fig, ax = plt.subplots()
                    ax.barh("Trigram", "Frequency", color='red', height=0.4, data=trigrams_neg_df)
                    st.write("Top 10 words in negative reviews - TRIGRAM ANALYSIS")
                    st.pyplot(fig)

    else:
        st.warning('Please upload the file in the required format')
