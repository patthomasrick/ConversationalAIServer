import json

from flask import Flask, current_app, request

from ChatServer.Backend.facebook_blenderbot_distill import FacebookBlenderbotDistill
from ChatServer.ml import _get_bot

app = Flask(__name__)
app.bot = _get_bot()


@app.route("/", methods=["POST", "DELETE"])
def index():
    headers = {"Content-Type": "application/json"}

    # If DELETE request, clear the chat history
    if request.method == "DELETE":
        if current_app.bot.clear_context():
            return json.dumps({"response": "Chat history cleared"}), 200, headers
        else:
            return json.dumps({"response": "Chat history not cleared"}), 500, headers

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
                "response": current_app.bot.tell(user_input),
            }
        ),
        200,
        headers,
    )
