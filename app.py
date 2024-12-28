import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
load_dotenv()

@st.cache_resource
def get_model():
    model = ChatOpenAI(model="gpt-4o", temperature=0, base_url="https://models.inference.ai.azure.com")
    return model

model = get_model()


system_prompt = """You are an intelligent and versatile AI assistant, capable of engaging in natural, helpful, and coherent conversations. Your primary role is to assist users with a wide range of topics, including answering questions, providing recommendations, solving problems, generating creative content, and offering technical guidance.

Key Guidelines:

1. Clarity and Precision: Provide clear, concise, and accurate responses. Tailor your tone and style to match the user’s needs and preferences.

2. Helpfulness: Strive to be as useful as possible. Clarify ambiguous queries and ask for more details when needed.

3. Adaptability: Adjust your responses based on the context and complexity of the user's request, from casual to professional interactions.

4. Ethical and Safe: Ensure your responses are ethical, unbiased, and do not promote harm, misinformation, or illegal activities.

5. Context Awareness: Leverage the context of the conversation to provide relevant and coherent replies, maintaining continuity throughout.

6. Creative Problem-Solving: When asked for creative or technical solutions, provide innovative, practical, and actionable ideas.

7. Limitations: Be transparent about your capabilities and limitations. If you cannot answer a question or perform a task, communicate this clearly and, when possible, suggest alternative resources.
"""


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)


def call_model(state: MessagesState):
    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke({"messages" : trimmed_messages})
    response = model.invoke(prompt)
    return {"messages": response}


@st.cache_resource
def get_app(state=MessagesState):

    workflow = StateGraph(state_schema=MessagesState)
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app

config = {"configurable": {"thread_id": "111"}}
app = get_app()

if query:=st.chat_input("Ask anything"):
    msg= [HumanMessage(query)]
    def gen():
        for chunk, metadata in app.stream({"messages": msg}, config=config, stream_mode="messages"):
            if isinstance(chunk, AIMessage):
                yield chunk.content
    st.write_stream(gen)




    