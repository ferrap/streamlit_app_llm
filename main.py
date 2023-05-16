"""Python file to serve as the frontend"""
import streamlit as st
import os
from streamlit_chat import message
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains.llm_math.base import LLMMathChain
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool, load_tools
from langchain import SQLDatabase, SQLDatabaseChain
from sqlalchemy import create_engine
from google.oauth2 import service_account
from google.cloud import bigquery

open_ai_key = st.secrets['OPENAI_API_KEY']

# From here down is all the StreamLit UI.
st.set_page_config(
    page_title="BigQuery + Langchain Demo", page_icon=":bird:"
)
st.header("BigQuery + Langchain Demo")
st.write(
    "👋 This is a demo of connecting large language models to Google BigQuery."
)
st.write(
    "🤖 The chatbot is built with LangChain (agents)."
)

st.write("Examples you can try:")
st.write("- What was the average number of transactions in January?")
st.write("- What was the highest fee paid?")
st.write("- What was the lowest fee paid?")

st.sidebar.title("Data Sources")

llm = OpenAI(temperature=0)

# Connect to Snowflake and build the chain
@st.experimental_singleton
def build_bq_chain():
    engine = create_engine(
        'bigquery://bigquery-public-data/crypto_ethereum', credentials_info=st.secrets['gcp_service_account']
        
    )

    sql_database = SQLDatabase(engine)

    st.sidebar.header("BigQuery database has been connected")
    st.sidebar.write(f"{sql_database.table_info}")

    db_chain = SQLDatabaseChain(llm=llm, database=sql_database)
    return db_chain





# BigQuery tool
db_chain = build_bq_chain()

tools = [
    Tool(
        name="BigQuery Transactions",
        func=lambda q: db_chain.run(q),
        description=f"Useful when you want to answer questions about ETH transactions, and gas price. The input to this tool should be a complete english sentence. ",
    )
]


# Initialize LangChain agent and chain

memory = ConversationBufferMemory(memory_key="chat_history")

agent_chain = initialize_agent(
    tools, llm, agent="zero-shot-react-description", verbose=True, memory=memory
)


def refresh_chain():
    """Refresh the chain variables.."""
    print("refreshing the chain")
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["buffer"] = []
    print("chain refreshed")


if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


user_input = st.text_input("You: ", placeholder="Hello!", key="input")

if user_input:
    output = agent_chain.run(input=user_input)
    # output = index.query(user_input).response
    if not st.session_state["past"]:
        st.session_state["past"] = []
    if not st.session_state["generated"]:
        st.session_state["generated"] = []
    st.session_state["past"].append(user_input)
    st.session_state["generated"].append(output)


if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
        message(st.session_state["generated"][i], key=str(i))


st.button("Clear chat", on_click=refresh_chain)

# if chain.memory.store:
#     for entity, summary in chain.memory.store.items():
#         # st.sidebar.write(f"{entity}: {summary}")
#         st.sidebar.write(f"Entity: {entity}")
#         st.sidebar.write(f"{summary}")
