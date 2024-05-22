# nltk.py

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# 必要なデータをダウンロード
nltk.download('punkt')
nltk.download('stopwords')

text = "Natural Language Processing with NLTK is interesting!"

# トークン化
tokens = word_tokenize(text)
print("Tokens:", tokens)

# ストップワードの除去
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print("Filtered Tokens:", filtered_tokens)
