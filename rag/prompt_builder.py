import config


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise and helpful assistant. Your job is to answer \
questions based strictly on the context documents provided below.

Rules you must follow:
- Answer ONLY using information found in the provided context.
- If the context does not contain enough information to answer, say clearly: \
"I don't have enough information in my knowledge base to answer that."
- Never invent facts, statistics, or details not present in the context.
- When you use information from a specific source, mention it naturally \
(e.g. "According to machine_learning.txt...").
- Keep answers clear, concise, and well-structured.
- If the question is a follow-up, use the conversation history to understand \
what the user is referring to."""


# ── Context formatter ─────────────────────────────────────────────────────────

def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a clearly labeled context block.

    Each chunk is separated and labeled with its source file so the
    LLM can reference it accurately in the answer.

    Args:
        chunks: List of dicts from retriever.retrieve()
                Each has: text, source, score

    Returns:
        A formatted multi-line string ready to embed in the prompt.

    Example output:
        [Source: machine_learning.txt | Relevance: 0.821]
        Machine learning is a subset of artificial intelligence...

        [Source: neural_nets.txt | Relevance: 0.743]
        Neural networks are computing systems inspired by...
    """
    if not chunks:
        return "No relevant context found in the knowledge base."

    parts = []
    for chunk in chunks:
        header = (
            f"[Source: {chunk['source']} | "
            f"Relevance: {chunk.get('score', 0):.2f}]"
        )
        parts.append(f"{header}\n{chunk['text']}")

    return "\n\n".join(parts)


# ── History formatter ─────────────────────────────────────────────────────────

def format_history(history: list[dict]) -> str:
    """
    Format chat history into a readable conversation transcript.

    Args:
        history: List of dicts from database.get_history()
                 Each has: role ('user' or 'assistant'), content

    Returns:
        A formatted conversation string, or empty string if no history.

    Example output:
        User: What is machine learning?
        Assistant: Machine learning is a field of AI that...

        User: What about deep learning?
        Assistant: Deep learning is a subset of machine learning...
    """
    if not history:
        return ""

    lines = []
    for msg in history:
        role    = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"].strip()
        lines.append(f"{role}: {content}")

    return "\n\n".join(lines)


# ── Main builder ──────────────────────────────────────────────────────────────

def build_prompt(
    chunks:  list[dict],
    history: list[dict],
    query:   str,
) -> str:
    """
    Assemble the complete prompt from all three inputs.

    Structure (in order):
      1. System role + grounding instruction
      2. Retrieved context with source labels
      3. Conversation history (if any)
      4. Current user question

    Args:
        chunks:  Retrieved and re-ranked chunks from retriever.retrieve()
        history: Past messages from database.get_history()
        query:   The current user question

    Returns:
        A single string ready to send to any LLM.
    """
    context_block  = format_context(chunks)
    history_block  = format_history(history)

    # Build the prompt section by section
    sections = [SYSTEM_PROMPT]

    sections.append(
        "--- CONTEXT DOCUMENTS ---\n" + context_block
    )

    if history_block:
        sections.append(
            "--- CONVERSATION HISTORY ---\n" + history_block
        )

    sections.append(
        "--- CURRENT QUESTION ---\n"
        f"Question: {query.strip()}\n\n"
        "Answer:"
    )

    prompt = "\n\n".join(sections)

    # Debug: log prompt length so you can tune chunk/history sizes
    print(
        f"[prompt_builder] Prompt assembled | "
        f"chunks={len(chunks)} | "
        f"history_turns={len(history)} | "
        f"total_chars={len(prompt)}"
    )

    return prompt