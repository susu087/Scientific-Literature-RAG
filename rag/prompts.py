"""Prompt and answer templates for the minimal RAG prototype."""

from typing import Dict, List


SYSTEM_PROMPT = (
    "You are a research assistant for scientific literature retrieval. "
    "Answer the user's question using only the provided context. "
    "When you rely on a document, cite it with its bracketed id such as [D1]. "
    "Only use citation ids that appear in the provided context. "
    "If the context is insufficient, say so explicitly."
)


ANSWER_INSTRUCTIONS = (
    "Write a concise answer grounded in the retrieved papers. "
    "Every key point must include at least one citation from the context. "
    "Use only citation ids that appear in the context, such as [D1] or [D2]. "
    "Do not cite papers that are not actually used in the answer. "
    "Do not invent paper details that are not present in the context. "
    "End with a 2-4 sentence summary, and the summary must also include at least one citation."
)


MOCK_ANSWER_TEMPLATE = (
    "Based on the retrieved literature, the most relevant evidence points to "
    "{summary} {citations}"
)


def build_answer_messages(question: str, context: str) -> List[Dict[str, str]]:
    """Build a future-proof chat prompt for a local/API generation backend."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Instructions:\n{ANSWER_INSTRUCTIONS}\n\n"
                f"Context:\n{context}"
            ),
        },
    ]
