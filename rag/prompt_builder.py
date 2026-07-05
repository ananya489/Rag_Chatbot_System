import config


# -------------------------------------------------
# SYSTEM PROMPT
# -------------------------------------------------

SYSTEM_PROMPT = """
You are a precise knowledge-base assistant.

Your job is to answer questions ONLY using the context documents provided below.

RULES:
1. Use only information from the provided context documents.
2. Do not use outside knowledge.
3. If the answer is not present in the context, say:
   "I don't have information about that in my knowledge base."
4. Never guess or create fake facts, numbers, quotes, or sources.
5. When answering, cite sources naturally:
   - With page number:
     (Source: filename, page N)
   - Without page number:
     (Source: filename)
6. For follow-up questions, use conversation history to understand references.
7. Keep answers clear and concise.
8. Never mention a page number unless it exists in the provided context.
"""


# -------------------------------------------------
# CONTEXT FORMATTER
# -------------------------------------------------

def format_context(chunks: list[dict]) -> str:
    """
    Convert retrieved chunks into LLM readable context.
    """

    if not chunks:
        return "No relevant context found in the knowledge base."


    parts = []


    for chunk in chunks:


        page_info = ""


        if chunk.get("page"):

            page_info = (
                f", page {chunk['page']}"
            )


        header = (
            f"[Source: {chunk['source']}"
            f"{page_info} | "
            f"Relevance: {chunk.get('score',0):.2f}]"
        )


        parts.append(
            f"{header}\n{chunk['text']}"
        )


    return "\n\n".join(parts)




# -------------------------------------------------
# HISTORY FORMATTER
# -------------------------------------------------

def format_history(history:list[dict]) -> str:


    if not history:

        return ""



    messages=[]



    for msg in history:


        role = (

            "User"

            if msg["role"]=="user"

            else

            "Assistant"

        )


        messages.append(

            f"{role}: {msg['content'].strip()}"

        )


    return "\n\n".join(messages)




# -------------------------------------------------
# PROMPT BUILDER
# -------------------------------------------------

def build_prompt(
    chunks:  list[dict],
    history: list[dict],
    query:   str,
    summary: str = "",          # ← new optional parameter
) -> str:
    """
    Assemble the complete prompt from all inputs.

    Structure (in order):
        1. System role + grounding instruction
        2. Retrieved context chunks with source labels
        3. Conversation summary (if present)
        4. Recent conversation turns (verbatim)
        5. Current user question + "Answer:" primer

    Args:
        chunks:  Retrieved and re-ranked chunks from retriever.
        history: Recent message turns to include verbatim.
        query:   The current user question.
        summary: Optional compressed summary of older turns.
                 Empty string means no summary section is added.

    Returns:
        A single formatted string ready to send to any LLM.
    """
    context_block = format_context(chunks)
    history_block = format_history(history)

    sections = [SYSTEM_PROMPT]

    sections.append("--- CONTEXT DOCUMENTS ---\n" + context_block)

    # Include summary before recent history if one exists
    if summary.strip():
        sections.append(
            "--- CONVERSATION SUMMARY ---\n"
            "The following is a summary of the earlier conversation:\n\n"
            + summary.strip()
        )

    if history_block:
        sections.append(
            "--- RECENT CONVERSATION ---\n" + history_block
        )

    sections.append(
        "--- CURRENT QUESTION ---\n"
        f"Question: {query.strip()}\n\n"
        "Answer:"
    )

    prompt = "\n\n".join(sections)

    print(
        f"[prompt_builder] Assembled | "
        f"chunks={len(chunks)} | "
        f"recent_turns={len(history)} | "
        f"has_summary={'yes' if summary.strip() else 'no'} | "
        f"total_chars={len(prompt)}"
    )

    return prompt




# -------------------------------------------------
# QUERY EXPANSION
# -------------------------------------------------

def expand_query(
        query:str,
        history:list[dict]
)->str:


    if not history or len(query.split()) > 8:

        return query



    vague_words={

        "it",
        "this",
        "that",
        "they",
        "them",
        "those",
        "the second",
        "the first",
        "the last",
        "what about"

    }



    query_lower=query.lower()



    if not any(
        word in query_lower
        for word in vague_words
    ):

        return query




    last_assistant = next(

        (

            m["content"]

            for m in reversed(history)

            if m["role"]=="assistant"

        ),

        ""

    )



    if last_assistant:


        first_sentence = (

            last_assistant
            .split(".")[0]

        )


        expanded=(

            f"{query} "
            f"(context: {first_sentence})"

        )


        print(

            f"[prompt_builder] "
            f"Query expanded: {expanded[:100]}"

        )


        return expanded



    return query