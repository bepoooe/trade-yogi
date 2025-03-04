import os
import google.generativeai as genai
from dotenv import load_dotenv


# Configure API Key
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))



def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    try:
        file = genai.upload_file(path, mime_type=mime_type)
        print(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file
    except Exception as e:
        print(f"File upload failed: {e}")
        return None

# Model configuration
generation_config = {
    "temperature": 0.15,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
}

try:
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        system_instruction=(
            "You are an expert at Trading insight of India. Your task is to engage in conversations about trade "
            "and answer questions. Explain specifically about the company so that users easily understand the background, "
            "history, and performance of that specific company in the trading market. Use humor and formality to make "
            "conversations educational and interesting. Ask questions to better understand the user and improve the "
            "educational experience. Suggest ways to relate these concepts to real-world observations and experiments."
        ),
    )
except Exception as e:
    print(f"Failed to configure model: {e}")
    exit()

# Optionally upload files
files = []
file_path = "file.txt"  # Update with an actual file path
uploaded_file = upload_to_gemini(file_path, mime_type="text/plain")
if uploaded_file:
    files.append(uploaded_file)

history = []
print("Hello, How can I help you?")

while True:
    try:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        chat_session = model.start_chat(history=history)
        response = chat_session.send_message(user_input)
        model_response = response.text

        print(f"Bot: {model_response}\n")

        history.append({"role": "user", "parts": [user_input]})
        history.append({"role": "model", "parts": [model_response]})
    except Exception as e:
        print(f"An error occurred: {e}")
        break
