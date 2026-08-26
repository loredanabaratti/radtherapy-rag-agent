from __future__ import annotations

from typing import Optional, TypedDict

import chromadb
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from sentence_transformers import SentenceTransformer

from src.config import (
    CHROMA_COLLECTION,
    CHROMA_DB_DIR,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    RETRIEVAL_K,
)

from src.schemas import AnswerOutput, CriticOutput, RouterDecision, SourceReference

class GraphState(TypedDict):
    """
    Shared state passed between all LangGraph nodes.
    """
    question: str
    router_decision: Optional[dict]
    retrieved_sources: list[dict]
    answer: Optional[dict]
    critic_result: Optional[dict]
    retry_count: int

def get_llm() -> ChatOllama:
    """
    Create the local Ollama chat model client.
    """
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
    )

def get_embedding_model() -> SentenceTransformer:
    """
    Load the local embedding model for document retrieval.
    """
    return SentenceTransformer(EMBEDDING_MODEL)

def get_collection() -> chromadb.Collection:
    """
    Open the persistent ChromaDB collection.
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    return client.get_collection(CHROMA_COLLECTION)

def router_node(state: GraphState) -> dict:
    """
    Decide whether to retrieve documents based on the question.
    """
    llm = get_llm().with_structured_output(RouterDecision)

    prompt = f"""
    You are a router that decides whether to retrieve documents for a given question.
    Question: {state['question']}
    
    Decide whether the user question requires information from the local document corpus. 
    Use retrieval from domain-specific questions about radiotherapy, treatment planning, dose distributions, IMRT, or DVHs.
    Do not use retrieval for simple conversation or general knowledge questions.
    """
    decision = llm.invoke(prompt)
    return {"router_decision": decision.model_dump()}

def retriever_node(state: GraphState) -> dict:
    """
    Retrieve relevant documents from the local ChromaDB collection based on the question.
    """
    router_decision = state["router_decision"] or {}

    if not router_decision.get("needs_retrieval", True):
        return {"retrieved_sources": []}
    search_query = router_decision.get("search_query", state["question"])

    embedding_model = get_embedding_model()
    query_embedding = embedding_model.encode(
        search_query, 
        normalize_embeddings=True
        ).tolist()

    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=RETRIEVAL_K,
        include=["documents", "metadatas", "distances"],
    )
    retrieved_sources = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        source = SourceReference(
            title=metadata.get("title", "Unknown"),
            source=metadata.get("source", "Unknown"),
            url=metadata.get("url", "Unknown"),
            chunk_index=metadata.get("chunk_index", -1),
        )
        retrieved_sources.append(
            {
                "content": document,
                "reference": source.model_dump(),
                "distance": distance,
            }
        )
    return {"retrieved_sources": retrieved_sources}

def answer_node(state: GraphState) -> dict:
    """
    Generate an answer to the question using the retrieved documents as context.
    """

    retrieved_sources = state.get("retrieved_sources", [])
    critic_feedback = (state.get("critic_result") or {}).get("feedback", "")

    if not retrieved_sources:
        answer = AnswerOutput(
            answer=(
                "I do not have enough information to answer your question based on the local document corpus."
            ),
            source_indices=[],
            insufficient_context=True,
        )
        return {"answer": answer.model_dump()}

    context_blocks = []

    for index, item in enumerate(retrieved_sources):
        reference = item["reference"]
        context_blocks.append(
            f"""[Source {index}]
Title: {reference.get("title", "Unknown")}
Source: {reference.get("source", "Unknown")}
URL: {reference.get("url", "Unknown")}

Content: {item["content"]}
"""
        )
    context = "\n\n".join(context_blocks)

    retry_instruction = ""
    if critic_feedback:
        retry_instruction = f"""
        A critic has provided feedback on the previous answer: {critic_feedback}
        Please consider this feedback when generating the new answer.
        """
        
    llm = get_llm().with_structured_output(AnswerOutput)

    prompt = f"""
    You are a knowledgeable assistant specialized in radiotherapy and treatment planning.
    Use the following context from local documents to answer the user's question.
    
    Provide a clear and concise answer. If the context is insufficient, indicate that you cannot provide a reliable answer.
    Answer the user's question using only the provided context sources. Do not use outside knowledge.
    Use concise, educational language. This is a technical demonstration and not clinical decision support. Do not provide medical advice or recommendations.

    For source_indices, list the zero-based indices of the sources that directly support your answer.
    {retry_instruction}

    User Question: 
    {state['question']}

    Context:
    {context}
    """
    answer = llm.invoke(prompt)
    return {"answer": answer.model_dump()}

def critic_node(state: GraphState) -> dict:
    """
    Evaluate the generated answer and provide feedback.
    """
    answer_data = state.get("answer")

    if not answer_data:
        raise ValueError("No answer data found in state for critic evaluation.")
    
    if answer_data["insufficient_context"]:
        critic = CriticOutput(
            is_grounded=True,
            feedback="The answer indicates insufficient context. Ensure that the retrieval process is working correctly and that relevant documents are available."
        )
        return {"critic_result": critic.model_dump()}

    retrieved_sources = state.get("retrieved_sources", [])
    answer_sources = answer_data.get("source_indices", [])

    context_blocks = []

    for index, item in enumerate(retrieved_sources):
        if index in answer_sources:
            reference = item["reference"]

            context_blocks.append(
                f"""[Source {index}]
Title: {reference.get("title", "Unknown")}
Content: {item["content"]}
"""
            )

    if not context_blocks:
        critic = CriticOutput(
            is_grounded=False,
            feedback=("No context blocks found for the sources cited in the answer. Rewrite the answer using the provided content and cite valid sources.")
        )
        return {"critic_result": critic.model_dump()}

    context_str = "\n".join(context_blocks)

    llm = get_llm().with_structured_output(CriticOutput)

    prompt = f"""
    You are a grounding critic that evaluates the answer provided by an assistant based on the context from local documents.
    Your task is to determine whether the answer is well-grounded in the provided context and to provide constructive feedback.

    Mark is_grounded as true only if:
    1. The answer makes no important claim that is unsupported by the provided context.
    2. The answer does not introduce outside knowledge
    3. The cited source indices are appropriate

    If unsupported claims exist, set is_grounded to false. In feedback, identify what must be removed, qualified, or corrected. Be concise and specific in your feedback.

    User Question:
    {state.get("question", "")}

    Assistant Answer to evaluate:
    {answer_data.get("answer", "")}

    Cited source excerpts from local documents:
    {context_str}
    """

    critic_result = llm.invoke(prompt)
    return {"critic_result": critic_result.model_dump()}

def increment_retry_node(state: GraphState) -> dict:
    """
    Increment the retry count in the state before generating a revised answer.
    """
    return {"retry_count": state.get("retry_count", 0) + 1}

def route_after_router(state: GraphState) -> str:
    """
    Determine the next node after the router based on the router's decision.
    Choose whether to retrieve documents or answer directly.
    """

    router_decision = state.get("router_decision", {})

    if router_decision.get("needs_retrieval", True):
        return "retrieve"
    
    return "answer"

def route_after_critic(state: GraphState) -> str:
    """
    Determine the next node after the critic based on the critic's evaluation.
    If the answer is grounded, end the process. If not, retry once.
    """

    critic_result = state.get("critic_result", {})
    retry_count = state.get("retry_count", 0)

    if not critic_result.get("is_grounded", False) and retry_count < 1:
        return "retry"
    
    return "end"

def build_graph():
    """
    Build the LangGraph state graph for the radiotherapy question-answering pipeline.
    """

    workflow = StateGraph(GraphState)

    workflow.add_node("router", router_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("answer", answer_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("increment_retry", increment_retry_node)

    #define the graph flow
    workflow.add_edge(START, "router")

    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "retrieve": "retriever",
            "answer": "answer",
        },
    )

    workflow.add_edge("retriever", "answer")
    workflow.add_edge("answer", "critic")

    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "retry": "increment_retry",
            "end": END,
        },
    )

    workflow.add_edge("increment_retry", "answer")

    return workflow.compile()