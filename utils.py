from langchain.memory import ConversationBufferMemory

# Initialize memory
def init_memory():
    return ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        input_key='question',  # Ensure the keys match your usage
        output_key='answer'
    )
MEMORY = init_memory()

