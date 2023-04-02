from ml import chat, initialize


if __name__ == "__main__":
    initialize(None, cli=True)

    while True:
        user_input = input(">>> ")

        if not user_input:
            break
        print(chat(user_input, cli=True))
