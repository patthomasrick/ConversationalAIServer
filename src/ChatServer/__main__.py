import logging
from ChatServer.ml import _get_bot
from datetime import datetime

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    logging.info("Loading model...")
    before = datetime.now()
    bot = _get_bot()
    after = datetime.now()
    logging.info(f"Done loading model in {after - before}. Enter nothing to quit.")

    while True:
        user_input = input(">>> ")

        if not user_input:
            logging.info("Received empty input. Quitting...")
            break

        before = datetime.now()
        response = bot.tell(user_input)
        after = datetime.now()

        logging.info(f"Took {after - before} to respond.")

        print(response)
