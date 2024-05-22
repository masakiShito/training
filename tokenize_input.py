from janome.tokenizer import Tokenizer
import sys

def tokenize_text(text):
    # Janomeのトークナイザーを使用してトークン化
    t = Tokenizer()
    tokens = [token.surface for token in t.tokenize(text)]
    return tokens

def main():
    # コマンドライン引数からテキストを取得
    if len(sys.argv) > 1:
        input_text = ' '.join(sys.argv[1:])
    else:
        print("テキストを入力してください。")
        sys.exit(1)

    # トークン化を実行
    tokens = tokenize_text(input_text)
    print("Tokens:", tokens)

if __name__ == "__main__":
    main()
