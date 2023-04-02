import json

from flask import Flask, request

from ml import chat, initialize, clear_history, get_history

app = Flask(__name__)

# Load models into memory.
initialize(app)


@app.route("/", methods=["POST", "DELETE"])
def index():
    headers = {"Content-Type": "application/json"}

    # If DELETE request, clear the chat history
    if request.method == "DELETE":
        clear_history()
        return json.dumps({"response": "Chat history cleared"}), 200, headers

    # Require a POST request
    if request.method != "POST":
        return json.dumps({"error": "no POST"}), 400, headers

    # Get the user input from the request
    user_input = request.form.get("user_input")
    if user_input is None:
        return json.dumps({"error": "no input"}), 400, headers

    return (
        json.dumps(
            {
                "input": user_input,
                "response": chat(user_input),
                "history": get_history(),
            }
        ),
        200,
        headers,
    )
