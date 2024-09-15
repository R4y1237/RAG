#import os and constants
import os
import constants

#set API keys
os.environ["OPENAI_API_KEY"] = constants.APIKEY
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")


#load csv data using DocumentLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.csv_loader import CSVLoader

#loader = CSVLoader(file_path="./demo.csv")
loader = DirectoryLoader('.', glob='./*.csv', loader_cls=CSVLoader)
data = loader.load()


#Split the data 
from langchain_text_splitters import RecursiveCharacterTextSplitter

#notice that there is a 200 chunk overlap. The overlap is to mitigate the possiblilty of seperating a statement from important context
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(data)


#Index and store data so that we can search on runtime
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# the embedding is the indexing part and chroma stores it
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())


#retrieves relevant data based on query. k represents the amount of docs you want to retrieve
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})



#Retrieval and Generation with Chatgpt

from langchain import hub
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

### Contextualize question ###
# the prompt is for the RAG to give context on what the AI is supposed to do
# You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
# If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
# Question: filler question 
# Context: filler context 
# Answer:

#this subchain is to make the llm use previous chat history to formulate the current question with context
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


### Answer question ###

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. Output as much as you can."
    "If you don't know the answer, ask for more details or say that you "
    "don't know. Try to give mutiple options. Keep the"
    "answer concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# dictionary to store message history
store = {}

# funciton to store messages
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


def get_response(msg):

    return conversational_rag_chain.invoke(
            {"input": msg},
            config={
            "configurable": {"session_id": "abc123"}
            },  # constructs a key "abc123" in `store`.
        )["answer"]



# final q&a loop to have conversations with the LLM
while True:
    question = input("You: ")

    # commands to quit session
    if question in ("","q","quit","e","end","exit"):
        break
    else:
        print(conversational_rag_chain.invoke(
            {"input": question},
            config={
            "configurable": {"session_id": "abc123"}
            },  # constructs a key "abc123" in `store`.
        )["answer"])




