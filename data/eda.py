import pandas as pd
import string
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))

df_train = pd.read_csv("training.csv", header=None, names=['id', 'text', 'label'])
df_test = pd.read_csv("test.csv", header=None, names=['id', 'text'])

def clean_text(text): 
    if pd.isnull(text): 
        return ""
    #making everything lowercase
    text = text.lower()
    # removing punctuation
    # THOUGHT: does this make sense? punctuation could convey information... 
    # but usually it only accentuates information that is already in the text.
    text = text.translate(str.maketrans('','', string.punctuation))
    text = text.strip()
    return text

df_train['clean'] = df_train['text'].apply(clean_text)
df_test['clean'] = df_test['text'].apply(clean_text)
df_train['word_count'] = df_train['text'].apply(lambda x: len(str(x).split()))
df_test['word_count'] = df_test['text'].apply(lambda x: len(str(x).split()))
def make_plot(df_train, df_test): 
    plt.figure(figsize=(10, 6))
    plt.hist(df_train['word_count'], bins=30, edgecolor='black')
    plt.title("Distribution of Review Lengths (in Words)")
    plt.xlabel("Word Count")
    plt.ylabel("Number of Reviews")
    plt.grid(True)
    plt.savefig("word_count_train.png")

    plt.figure(figsize=(10, 6))
    plt.hist(df_test['word_count'], bins=30, edgecolor='black')
    plt.title("Distribution of Review Lengths (in Words)")
    plt.xlabel("Word Count")
    plt.ylabel("Number of Reviews")
    plt.grid(True)
    plt.savefig("word_count_test.png")
    #saving the df in new files
    df_train.to_csv("cleaned_train.csv", index=False)
    df_test.to_csv("cleaned_test.csv", index=False)

# what i want to do: 
# basic data cleaning: all lowercase, remove punctuation, special characters, remove stopwords (done)

# inspect the sequence length of training and testing data (done)

# Word frequency analysis
all_words = []
def count_words(df): 
    for review in df:
        if pd.isnull(review):
            continue
        words = review.lower().split()
        words = [word.strip('.,!?";:()[]') for word in words]
        words = [word for word in words if word not in stopwords]
        all_words.extend(words)

    word_cntr = Counter(all_words)
    print(word_cntr.most_common(20))

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_cntr)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("wordcloud_negative.png", dpi=300)
    
    # for review in df_test['clean']:
    #     if pd.isnull(review):
    #         continue
    #     words = review.lower().split()
    #     words = [word.strip('.,!?";:()[]') for word in words]
    #     words = [word for word in words if word not in stopwords]
    #     all_words.extend(words)     

    # word_cntr = Counter(all_words)
    # print(word_cntr.most_common(20))

    # wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_cntr)
    # # Word cloud !!! looks very cool (done)
    # plt.figure(figsize=(10, 5))
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis('off')
    # plt.tight_layout()
    # plt.savefig("wordcloud_test.png", dpi=300)
#count_words(df_train['clean'])

def sentiment_dist(df_train, df_test): 
    print(df_train['label'].value_counts())
    print(df_train.groupby('label')['word_count'].mean())

sentiment_dist(df_train, df_test)
# output: 
# neutral     49148
# positive    31039
# negative    21910
# mean amount of words in each sentiment:
# negative    15.556686
# neutral     12.427891
# positive    14.372499


# Sentiment distribution => analyze the amount of neutral/ negative / postive labels 

# TF-IDF scores => what words are important / unique?
# Term frequency - Inverse Document Frequency: a term is important if it is frequent across this document, and not frequent across another document.
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df_train['clean'])
words = vectorizer.get_feature_names_out()
first_review_scores = tfidf_matrix[0].T.todense()
df_tfidf = pd.DataFrame(first_review_scores, index=words, columns=["tfidf"])
top_words = df_tfidf.sort_values(by="tfidf", ascending=False).head(10)
print(top_words)

positive_reviews = df_train[df_train['label'] == 'positive']['clean']
neutral_reviews = df_train[df_train['label'] == 'neutral']['clean']
negative_reviews = df_train[df_train['label'] == 'negative']['clean']

#count_words(positive_reviews)
#count_words(neutral_reviews)
#count_words(negative_reviews)

# N-gram analysis (bigrams: not good, very tasty)

def get_top_ngrams(text, ngram_range=(2,2), top_n = 10): 
    vec = CountVectorizer(ngram_range = ngram_range, stop_words='english')
    X = vec.fit_transform(text)
    ngram_counts = X.sum(axis=0).A1
    ngram_names = vec.get_feature_names_out()
    ngram_freq = list(zip(ngram_names, ngram_counts))
    sorted_ngrams = sorted(ngram_freq, key=lambda x: x[1], reverse = True)
    return pd.DataFrame(sorted_ngrams[:top_n], columns=['ngram', 'count'])

top_pos_bigrams = get_top_ngrams(positive_reviews, ngram_range=(2, 2), top_n=10)
top_neu_bigrams = get_top_ngrams(neutral_reviews, ngram_range=(2, 2), top_n=10)
top_neg_bigrams = get_top_ngrams(negative_reviews, ngram_range=(2, 2), top_n=10)
def plot_ngrams(df, title, color='skyblue'):
    plt.figure(figsize=(8, 4))
    plt.barh(df['ngram'], df['count'], color=color)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig("bi_grams_neg.png", dpi=300)
    plt.show()

#plot_ngrams(top_pos_bigrams, "Top Bigrams in Positive Reviews", "#8BC34A")
#plot_ngrams(top_neu_bigrams, "Top Bigrams in Neutral Reviews", "#FFC107")
plot_ngrams(top_neg_bigrams, "Top Bigrams in Negative Reviews", "#F44336")