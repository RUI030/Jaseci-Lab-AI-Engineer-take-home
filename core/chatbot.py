class Chatbot:
    def ask(self, prompt: str) -> str:
        self.display(prompt)
        try:
            return input("> ").strip()
        except EOFError:
            return ""

    def display(self, message: str) -> None:
        print(message)
