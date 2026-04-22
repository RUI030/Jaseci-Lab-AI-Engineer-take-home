class Chatbot:
    def ask(self, prompt: str) -> str:
        self.display(prompt)
        return input("> ").strip()

    def display(self, message: str) -> None:
        print(message)
