from .blenderbot import TalkativeBlenderbot


class FacebookBlenderbotDistill(TalkativeBlenderbot):
    def __init__(self):
        super().__init__("facebook/blenderbot-400M-distill")

    def tell(self, message: str) -> str:
        output = super().tell(message)

        # Remove <s> and </s> tokens.
        return output.replace("<s>", "").replace("</s>", "").strip()
