#Streamlit is an open-source Python framework that lets you build interactive web apps 
#for data science and AI—without needing frontend skills like HTML, CSS, or JavaScript.
#We will add web based UI using streamlit

#We will first install streamlit here
#pip install streamlit

#Create a web interface
import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

#Load AI Model
llm = OllamaLLM(model = "mistral")

#Initialize memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory() #Stores User-AI Conversation history

#Define AI chat prompt
prompt = PromptTemplate(
  input_variables=["chat_history","question"],
  template="Previous conversation: {chat_history}\nUser: {question}\nAI:"
)

#Function to run AI Chat with memory
#This defines a function that processes the chat 
def run_chain(question):
    # Use session_state everywhere
    history = st.session_state.chat_history
        
    #retrieve chat history manually
    chat_history_text = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in history.messages])

    #Run the AI response generation
    #This will send the formatted chat history and user question to the AI model
    response = llm.invoke(prompt.format(chat_history=chat_history_text,question=question))
    
    #Store new user input and AI response in the memory
    history.add_user_message(question)
    history.add_ai_message(response)

    #returns the AI generated response to be displayed in the chat
    return response

#Streamlit UI
st.title("🤖 Smart AI Assistant")
st.write("💬 I'm here to help — ask me anything!")

user_input = st.text_input("💬 Your Question")

if user_input:
    response = run_chain(user_input)
    st.write(f"**You:** {user_input}")
    st.write(f"**AI:** {response}")

#Show full chat
st.subheader("📜 Chat History")
for msg in st.session_state.chat_history.messages:
    st.write(f"**{msg.type.capitalize()}**: {msg.content}")

