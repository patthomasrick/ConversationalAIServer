from ChatServer.Backend.facebook_blenderbot_distill import FacebookBlenderbotDistill


def _get_bot():
    bot = FacebookBlenderbotDistill()
    bot.load()
    return bot


if __name__ == "__main__":
    _get_bot()
