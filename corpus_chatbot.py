from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# コーパスの読み込み
def load_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        corpus = file.readlines()
    return [line.strip() for line in corpus]

# Janomeのトークナイザーを初期化
tokenizer = Tokenizer()

# トークン化関数の定義
def tokenize(text):
    return [token.surface for token in tokenizer.tokenize(text)]

# ユーザー入力に基づく応答生成
def get_response(user_input, corpus, vectorizer, corpus_vectors):
    # ユーザー入力をベクトル化
    user_input_vector = vectorizer.transform([user_input])
    
    # コサイン類似度の計算
    similarities = cosine_similarity(user_input_vector, corpus_vectors)
    
    # 最も類似度の高い文を応答として選択
    best_match_index = similarities.argmax()
    return corpus[best_match_index]

def main():
    print("チャットボットにようこそ！'さようなら'と入力すると終了します。")

    # コーパスの読み込み
    corpus = load_corpus('corpus.txt')

    # TF-IDFベクトライザーの初期化とコーパスのベクトル化
    vectorizer = TfidfVectorizer(tokenizer=tokenize)
    corpus_vectors = vectorizer.fit_transform(corpus)

    while True:
        # ユーザーからの入力を受け取る
        user_input = input("あなた: ")

        # 'さようなら'と入力されたら終了
        if user_input == "さようなら":
            print("チャットボット: さようなら！またお話ししましょう。")
            break

        # 応答を取得して表示する
        response = get_response(user_input, corpus, vectorizer, corpus_vectors)
        print("チャットボット:", response)

if __name__ == "__main__":
    main()
