from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# モデルとトークナイザーの読み込み
model_name = "rinna/japanese-gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name)

# チャットの履歴を保持するための変数
chat_history_ids = None

print("Chatbot: こんにちは！何かお話ししましょう。")

while True:
    # ユーザーからの入力を受け取る
    user_input = input("あなた: ")

    # 'さようなら'と入力されたら終了
    if user_input.lower() in ["さようなら", "さよなら", "バイバイ"]:
        print("Chatbot: さようなら！またお話ししましょう。")
        break

    # ユーザー入力をトークン化
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # モデルに入力を与えて応答を生成
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=100,  # 応答の最大長を制限
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        num_return_sequences=1,
        temperature=0.7,  # 生成の多様性を制御
        top_k=50,        # top_kサンプリングを使用
        top_p=0.9,       # top_pサンプリングを使用
    )

    # モデルの応答をデコードして表示
    bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print(f"Chatbot: {bot_response}")
