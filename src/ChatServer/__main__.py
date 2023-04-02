from ChatServer.ml import _get_bot


if __name__ == "__main__":
    bot = _get_bot()

    while True:
        user_input = input(">>> ")

        if not user_input:
            break
        print(bot.tell(user_input))
