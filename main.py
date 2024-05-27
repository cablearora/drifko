import os
import logging
from flask import Flask, render_template, jsonify, request
# from pymongo import MongoClient
# from pymongo.server_api import ServerApi
from langchain_community.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from chat_with_documents import configure_retrieval_chain
from prompt import user_prompt
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize the HuggingFaceHub LLM
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    huggingfacehub_api_token="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",
    model_kwargs={
        "max_new_tokens": 2000,
        "top_k": 30,
        "temperature": 0.1,
        "repetition_penalty": 1.03,
    },
)

# Initialize the memory buffer
MEMORY = ConversationBufferMemory(
    memory_key='chat_history',
    return_messages=True,
    input_key='input',
    output_key='output'
)

# MongoDB setup
# uri = "mongodb+srv://Drifko:PlYQdq1J9rJmBhfI@drifko.n3o7kkj.mongodb.net/?retryWrites=true&w=majority&appName=Drifko"
# client = MongoClient(uri, server_api=ServerApi('1'))
# db = client['drifko']

# Flask app setup
app = Flask(__name__)

@app.route("/")
def home():
    # chats = db.chats.find()
    # myChats = [chat for chat in chats]
    return render_template("index.html", )#myChats=myChats)

@app.route("/api", methods=["GET", "POST"])
def qa():
    if request.method == "POST":
        question = request.json.get("msg")
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        print(question)
        
        # Check for existing chat in the database
        # chat = db.chats.find_one({"question": question})
        
        
        # if chat:
        #     data = {"question": question, "answer": chat['answer']}
        #     return jsonify(data)
        
        # Use memory to keep track of conversation history
        MEMORY.save_context({"input": question}, {"output": ""})
        chat_history = MEMORY.load_memory_variables({})
        
        # Configure the retrieval chain with the prompt
        CONV_CHAIN = configure_retrieval_chain(prompt=user_prompt)
        
        # Generate response
        response = CONV_CHAIN.run({
            "question": question,
            "chat_history": chat_history
        })
        
        # Update memory with the response
        MEMORY.save_context({"input": question}, {"output": response})
        
        data = {"question": question, "answer": response}
        # db.chats.insert_one({"question": question, "answer": response})
        print(data)
        return jsonify(data)
    
    # Default response for GET requests
    data = {"result": "Thank you! I'm just a machine learning model designed to respond to questions and generate text based on my training data. Is there anything specific you'd like to ask or discuss?"}
    return print(parser.invoke(data))

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
