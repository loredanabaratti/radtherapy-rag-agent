from pydantic import BaseModel, Field

class RouterDecision(BaseModel):
    """
    Represents a decision made by the router to select a specific tool for processing.
    """
    needs_retrieval: bool = Field(
        description="Indicates whether the question requires retrieval from the document corpus."
        )
    search_query: str = Field(
        description="The search query to be used for retrieval if needs_retrieval is True. Empty if retrieval is not needed."
    )
    reasoning: str = Field(
        description="The reasoning behind the router's decision, explaining why retrieval is or isn't necessary."
    )

class SourceReference(BaseModel):
    """
    Source metadata returned with an answer.
    """

    title: str
    source: str
    url: str
    chunk_index: int

class AnswerOutput(BaseModel):
    """
    Grounded answer produced by the answer node.
    """

    answer: str = Field(
        description="The generated answer to the user's question based on the provided context."
    )
    source_indices: list[int] = Field(
        description="Zero-based indices of context sources in the answer."
    )
    insufficient_context: bool = Field(
        description="True when the provided context is insufficient to answer safely."
    )

class CriticOutput(BaseModel):
    """
    Grounding assessment produced by the critic node.
    """

    is_grounded: bool = Field(
        description="True if the answer is grounded in the provided context, False otherwise."
    )
    feedback: str = Field(
        description="Short explanation or correction instruction for the answer node."
    )
    