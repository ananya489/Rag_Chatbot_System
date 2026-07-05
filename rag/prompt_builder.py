from logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a precise knowledge-base assistant.

Your job is to answer questions ONLY using the context documents provided below.

RULES:
1. Use ONLY information from the provided context documents.
2. Do NOT use outside knowledge.
3. If the answer is not present in the context, say:
   "I don't have information about that in my knowledge base."
4. Never invent facts, numbers, quotations, or sources.
5. Cite sources naturally:
   - (Source: filename)
   - (Source: filename, page N)
6. Use recent conversation only to understand follow-up questions.
7. Keep answers concise and accurate.
8. Never mention a page number unless it exists in the retrieved context.
""".strip()


# ---------------------------------------------------------------------
# CONTEXT FORMATTER
# ---------------------------------------------------------------------

def format_context(chunks: list[dict]) -> str:
    """
    Convert retrieved chunks into readable context for the LLM.
    """

    if not chunks:
        return "No relevant context found in the knowledge base."

    sections = []

    for chunk in chunks:
        score = chunk.get("score", 0)

        if chunk.get("page"):
            header = (
                f"[Source: {chunk['source']} | "
                f"Page: {chunk['page']} | "
                f"Relevance: {score:.2f}]"
            )
        else:
            header = (
                f"[Source: {chunk['source']} | "
                f"Relevance: {score:.2f}]"
            )

        sections.append(f"{header}\n{chunk['text']}")

    return "\n\n".join(sections)


# ---------------------------------------------------------------------
# HISTORY FORMATTER
# ---------------------------------------------------------------------

def format_history(history: list[dict]) -> str:
    """
    Format conversation history.
    """

    if not history:
        return ""

    formatted = []

    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {msg['content'].strip()}")

    return "\n\n".join(formatted)


# ---------------------------------------------------------------------
# PROMPT BUILDER
# ---------------------------------------------------------------------

def build_prompt(
    chunks: list[dict],
    history: list[dict],
    query: str,
    summary: str = "",
) -> str:
    """
    Assemble the complete prompt.

    Order:
        System Prompt
        Context
        Summary (optional)
        Recent History (optional)
        Current Question
    """

    context = format_context(chunks)
    history_text = format_history(history)

    prompt_parts = [
        SYSTEM_PROMPT,
        f"--- CONTEXT DOCUMENTS ---\n{context}",
    ]

    if summary.strip():
        prompt_parts.append(
            "--- CONVERSATION SUMMARY ---\n"
            + summary.strip()
        )

    if history_text:
        prompt_parts.append(
            "--- RECENT CONVERSATION ---\n"
            + history_text
        )

    prompt_parts.append(
        "--- CURRENT QUESTION ---\n"
        f"Question: {query.strip()}\n\n"
        "Answer:"
    )

    prompt = "\n\n".join(prompt_parts)

    log.info(
        "Prompt assembled | chunks=%d | history=%d | summary=%s | chars=%d",
        len(chunks),
        len(history),
        "yes" if summary.strip() else "no",
        len(prompt),
    )

    return prompt