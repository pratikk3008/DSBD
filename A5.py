import random

class SimpleChatbot:
    def __init__(self):
        # Define responses for different types of inputs
        self.responses = {
            "greeting": ["Hello! How can I assist you today?", "Hi there! How can I help you?", "Hey! What can I do for you?"],
            "farewell": ["Goodbye! Have a great day!", "See you later!", "Farewell! Until next time!"],
            "default": ["I'm sorry, I don't understand that.", "Could you please rephrase your question?", "I'm not sure how to respond to that."]
        }

    def get_response(self, user_input):
        # Convert user input to lowercase for case-insensitive matching
        user_input = user_input.lower().strip()
        
        # Check for greetings
        if any(greet in user_input for greet in ["hi", "hello", "hey", "hola"]):
            return random.choice(self.responses["greeting"])
        
        # Check for farewells
        elif any(farewell in user_input for farewell in ["bye", "goodbye", "see you", "farewell"]):
            return random.choice(self.responses["farewell"])
        
        # Default response
        else:
            return random.choice(self.responses["default"])

def main():
    bot = SimpleChatbot()
    print("Chatbot: Hello! I'm here to chat with you. Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye! Have a great day!")
            break
        response = bot.get_response(user_input)
        print(f"Chatbot: {response}")

main()