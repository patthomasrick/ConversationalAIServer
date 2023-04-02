from .blenderbot import TalkativeBlenderbot


class FacebookBlenderbot3B(TalkativeBlenderbot):
    def __init__(self):
        super().__init__("facebook/blenderbot-3B")

    def tell(self, message: str) -> str:
        output = super().tell(message)

        # Remove <s> and </s> tokens.
        return output.replace("<s>", "").replace("</s>", "").strip()
