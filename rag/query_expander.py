def expand_query(query, history):

    """
    Improves user query for vector retrieval.
    """

    if history:

        last_messages = history[-2:]

        context = " ".join(
            [
                msg["content"]
                for msg in last_messages
            ]
        )

        return f"""
        User question:
        {query}

        Previous context:
        {context}

        Find relevant information about this topic.
        """

    return query