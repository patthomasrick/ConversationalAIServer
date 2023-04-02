from abc import ABC, abstractmethod


class Talkative(ABC):
    @abstractmethod
    def tell(self, message: str) -> str:
        """Tell the model a message and get the response.

        Args:
            message (str): Message to tell the model.

        Returns:
            str: Response from the model.
        """
        pass

    @abstractmethod
    def clear_context(self) -> bool:
        """Clear the context of the chat.

        Only applicable if the model is contextual.

        Returns:
            bool: True if the context was cleared, False otherwise.
        """
        pass
