from ChatServer.Backend.facebook_blenderbot_3b import FacebookBlenderbot3B


def _get_bot():
    bot = FacebookBlenderbot3B()
    bot.load()
    return bot


if __name__ == "__main__":
    _get_bot()
