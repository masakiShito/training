from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# コーパスの読み込み
def load_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        corpus = file.readlines()
    qa_pairs = [line.strip().split('\\') for line in corpus]
    return qa_pairs

# Janomeのトークナイザーを初期化
tokenizer = Tokenizer()

# トークン化関数の定義
def tokenize(text):
    return [token.surface for token in tokenizer.tokenize(text)]

# ユーザー入力に基づく応答生成
def get_response(user_input, questions, answers, vectorizer, question_vectors):
    # ユーザー入力をベクトル化
    user_input_vector = vectorizer.transform([user_input])
    
    # コサイン類似度の計算
    similarities = cosine_similarity(user_input_vector, question_vectors)
    
    # 最も類似度の高い質問を応答として選択
    best_match_index = similarities.argmax()
    
    # 類似度のログ出力
    print(f"入力: {user_input}")
    for i, question in enumerate(questions):
        print(f"質問: {question} 類似度: {similarities[0][i]}")
    print(f"最も類似した質問: {questions[best_match_index]} 類似度: {similarities[0][best_match_index]}")
    
    # 類似度が0.5未満の場合は適当な返答を選択する
    if similarities[0][best_match_index] < 0.5:
        return "すみません、よくわかりませんでした。"
    return answers[best_match_index]

def main():
    print("チャットボットにようこそ！'さようなら'と入力すると終了します。")

    # コーパスの読み込み
    qa_pairs = load_corpus('qa_corpus.txt')
    questions = [pair[0] for pair in qa_pairs]
    answers = [pair[1] for pair in qa_pairs]

    # TF-IDFベクトライザーの初期化と質問のベクトル化
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        vectorizer = TfidfVectorizer(tokenizer=tokenize)
        question_vectors = vectorizer.fit_transform(questions)

    while True:
        # ユーザーからの入力を受け取る
        user_input = input("あなた: ")

        # 'さようなら'と入力されたら終了
        if user_input == "さようなら":
            print("チャットボット: さようなら！またお話ししましょう。")
            break

        # 応答を取得して表示する
        response = get_response(user_input, questions, answers, vectorizer, question_vectors)
        print("チャットボット:", response)

if __name__ == "__main__":
    main()
