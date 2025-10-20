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
from tenacity import retry, stop_after_attempt, wait_fixed


import operator
from typing_extensions import TypedDict, Annotated
from typing import Literal
from pprint import pprint


# variables
from variables import elon_musk_info, summary_template
import logging
import json

load_dotenv()


# ------------------------------------------------------------IMPORTS------------------------------------------------------------


llm = ChatOllama(
    model="gpt-oss:20b",
    temperature=0,
    # other params...
)

def overwrite_merge(old, new):
    """Always prefer the new value if provided, else keep the old one."""
    return new if new else old


# Define state to hold messages and LLM calls count
class MultiAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    plan: Annotated[str, overwrite_merge]  # the planner's plan
    code: Annotated[str, overwrite_merge]  # the coder’s output
    review: Annotated[dict, overwrite_merge]  # the reviewer’s feedback
    llm_calls: int


# Reusable retry decorator for robustness
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def safe_invoke(llm, messages):
    return llm.invoke(messages)


# Create specialized LLMs
planner_llm = ChatOllama(
    model="gpt-oss:20b",
    temperature=0.3,
    # other params...
)
coder_llm = ChatOllama(
    model="gpt-oss:20b",
    temperature=0.4,
    # other params...
)
reviewer_llm = ChatOllama(
    model="gpt-oss:20b",
    temperature=0.1,
    # other params...
)


def planner_node(state: MultiAgentState):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a Planner. Read the user's request and create a short, clear plan describing what to code.",
            ),
            MessagesPlaceholder("messages"),
        ]
    )
    messages = prompt.invoke({"messages": state["messages"]}).to_messages()
    output = safe_invoke(planner_llm, messages)
    plan_text = output.content.strip()

    print(f"🧩 PLAN:\n{plan_text}\n")

    return {
        "messages": [output],
        "plan": plan_text,
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def coder_node(state: MultiAgentState):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a Coder. Follow the given plan and write working Python or java code based on the given plan.",
            ),
            ("user", "Plan: {plan}"),
        ]
    )
    output = safe_invoke(coder_llm, prompt.invoke({"plan" : state['plan']}).to_messages())

    code_text = output.content.strip()
    print(f"💻 CODE GENERATED:\n{code_text}\n")

    return {
        "messages": [output],
        "code": code_text,
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def reviewer_node(state: MultiAgentState):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a Reviewer. Review the following Python code for correctness and clarity.  "
                "Always respond strictly in the following JSON format but do not mention anything like `json` in the output:\n"
                "{{\n"
                '  "review_result": <overall review of the code>,\n'
                '  "regenerate_code": "<yes or no , yes if you want to regenarate the code no if it is not mandatory>",\n'
                "}}",
            ),
            ("user", "Gnerated Code:{gencode}"),
        ]
    )
    prompt_with_filled_value = prompt.invoke({"gencode": state["code"]}).to_messages()
    output = safe_invoke(reviewer_llm, prompt_with_filled_value)

    review_text = output.content.strip()
    print(f"🧾 REVIEW:\n{review_text}\n")
    
    updated_state = {
        "messages": [output],
        "review": output.content,
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


    return updated_state


def build_multi_agent_graph():
    graph = StateGraph(MultiAgentState)

    graph.add_node("planner_node", planner_node)
    graph.add_node("coder_node", coder_node)
    graph.add_node("reviewer_node", reviewer_node)

    graph.add_edge(START, "planner_node")
    graph.add_edge("planner_node", "coder_node")

    # Conditional edge for reviewer feedback
    def review_condition(state: MultiAgentState):
        review = state.get("review", "")
        try:
            review_dict = json.loads(review)
            if review_dict['regenerate_code'].lower() == "yes": # if the review code says to regenerate code
                return "coder_node"
            return END
        except Exception as e:
            print(e)
            return "reviewer_node"

    graph.add_conditional_edges(
        "coder_node", lambda s: "reviewer_node", ["reviewer_node"]
    )
    graph.add_conditional_edges("reviewer_node", review_condition, ["coder_node", "reviewer_node", END])

    return graph.compile()


def multi_agent_chatbot():
    
    initial_state = {
        "messages": [],
        "plan": "",
        "code": "",
        "review": "",
        "llm_calls": 0
	}
    
    
    while True:
        question = input("enter your question => ")
        if question.lower() in ["exit", "quit"]:
            break
        # Add new human message to the existing conversation
        initial_state["messages"].append(HumanMessage(question))
        agent = build_multi_agent_graph()
        final_state = agent.invoke(initial_state)
        print("**"*200)
        pprint(final_state['code'])
 
 
    # while True:
    #     question = input("enter your question => ")
    #     if question.lower() in ["exit", "quit"]:
    #         break

    #     # Add new human message to the existing conversation
    #     state["messages"].append(HumanMessage(question))
    #     state["memory"].append(HumanMessage(question).content)

    #     # Invoke the agent with the current state
    #     state = agent.invoke(state)

    #     # Extract and show the latest AI message
    #     last_msg = state["messages"][-1]
    #     print("🤖:", getattr(last_msg, "content", last_msg))


if __name__ == "__main__":
    print(os.environ.get("OPENAI_API_KEY"))
    multi_agent_chatbot()
