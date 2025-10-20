from dotenv import load_dotenv
import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    PromptTemplate,
    MessagesPlaceholder,
    ChatPromptTemplate,
)

from langchain_tavily import TavilySearch
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AnyMessage
import operator

from typing_extensions import TypedDict, Annotated
from typing import Literal
from pprint import pprint


# variables
from variables import elon_musk_info, summary_template
import logging

load_dotenv()


# ------------------------------------------------------------IMPORTS------------------------------------------------------------


llm = ChatOllama(
    model="gpt-oss:20b",
    temperature=0,
    # other params...
)

tool_search = TavilySearch(
    max_results=1,
    topic="general",
    # include_answer=False,
    # include_raw_content=False,
    # include_images=False,
    # include_image_descriptions=False,
    # search_depth="basic",
    # time_range="day",
    # include_domains=None,
    # exclude_domains=None
)

# ------------------------------------------------------------Extras------------------------------------------------------------


# ------------------------------------------------------------ TOOLS ------------------------------------------------------------
@tool
def add_three_numbers(a: int, b: int, c: int) -> int:
    """Add three numbers."""
    return a + b + c


@tool
def multiply_two_numbers(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b
# ------------------------------------------------------------------------------------------------------------------------



prev_chats = []


def simple_llmcall():

    # ***************************** [chat with memory] ****************************

    while True:
        user_question = input("Enter your query =>")

        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a general purpose llm trained to answer user question as best your knowledge. You have access to all the previoius chats {chats}",
                ),
                ("human", "{user_question}"),
            ]
        )
        # Format the prompt with the actual input data
        messages_to_send = prompt_template.invoke(
            {"user_question": user_question, "chats": prev_chats}
        ).to_messages()
        # Call the language model with formatted messages
        response = llm.invoke(messages_to_send)
        print(response.content)
        prev_chats.append(("human", user_question))
        prev_chats.append(("system", response.content))

    # *********************************************************
    # prompt_template = ChatPromptTemplate.from_messages([
    # ("system", "Given the following information:, give me a summary of the information."), ("human" , "The givn info is {elon_musk_info}")
    # ])
    # # Format the prompt with the actual input data
    # messages_to_send = prompt_template.invoke({"elon_musk_info": elon_musk_info}).to_messages()
    # # Call the language model with formatted messages
    # response = llm.invoke(messages_to_send)

    # *********************************************************
    # prompt = PromptTemplate(input_variables = ['information'], template=summary_template)
    # chain = prompt | llm
    # response = chain.invoke({"information": elon_musk_info})
    # print(response.content)

    # messages_to_send = [SystemMessage(content=f"given the follwoing information {elon_musk_info}, give me a summary of the information.")]
    # response = llm.invoke(messages_to_send)

    print(response.content)
    print("SUCCESS" * 10)



tools = [add_three_numbers, multiply_two_numbers, tool_search]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)


# Define state to hold messages and LLM calls count
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    memory: Annotated[list[str], operator.add]  # 🧠 Add this


# LLM call node: decides which tool to call based on input message
def llm_call(state: dict):
    print(f"llm_call =>", end="")
    pprint(state)
    print("**" * 100)
    # Define ChatPromptTemplate
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant that either adds three numbers or multiplies two numbers based on user input. "
                "If the input doesn't match these categories, use the Tavily tool for a global search. "
                "Always respond strictly in the following JSON format:\n"
                "{{\n"
                '  "operation": "<add/multiply/search>",\n'
                '  "result": <result or the the final answer if search>,\n'
                '  "search_query": "<query string or empty>",\n'
                '  "Source" : "<url of the search website you used for the answer>"\n'
                "}}",
            ),
            MessagesPlaceholder(
                "msgs"
            ),  # Placeholder for inserted dynamic conversation messages
        ]
    )

    messages_to_send = prompt_template.invoke({"msgs": state["messages"]}).to_messages()
    llm_output = llm_with_tools.invoke(messages_to_send)

    return {
        "messages": [llm_output],
        "llm_calls": state.get("llm_calls", 0) + 1,
        "memory": [llm_output.content],
    }


# Tool call node: Executes the tool calls requested by LLM
def tool_node(state: dict):
    print(f"tool_node => ", end="")
    pprint(state)
    print("**" * 100)

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


# Conditional function to continue loop if tool call was made, otherwise stop
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    print(f"should_continue => ", end="")
    pprint(state)
    print("**" * 100)

    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return END


def search_with_agent():
    # Build agent graph
    agent_builder = StateGraph(MessagesState)
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    agent_builder.add_edge("tool_node", "llm_call")

    # Compile agent
    agent = agent_builder.compile()
    # Example invocation: add three numbers
    # messages = [("human","give me some proof that earth is not flat")]
    #     messages = [("human","What is the sum of 765, 453 and 987")]
    #     messages = [("human","Give me the scorecard of latest ind vs aus cricket series")]
    # messages = [("human","Give me the schedule of messi coming to india for tour")]

    state = {"messages": [], "llm_calls": 0, "memory": []}
    while True:
        question = input("enter your question => ")
        if question.lower() in ["exit", "quit"]:
            break

        # Add new human message to the existing conversation
        state["messages"].append(HumanMessage(question))
        state["memory"].append(HumanMessage(question).content)

        # Invoke the agent with the current state
        state = agent.invoke(state)

        # Extract and show the latest AI message
        last_msg = state["messages"][-1]
        print("🤖:", getattr(last_msg, "content", last_msg))


if __name__ == "__main__":
    print(os.environ.get("OPENAI_API_KEY"))
    # simple_llmcall()
    search_with_agent()
