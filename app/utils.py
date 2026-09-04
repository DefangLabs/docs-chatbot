import sys
import traceback

from segment import analytics


# Shared function to generate response stream from RAG system
def generate(rag, query, source, anonymous_id):
    full_response = ""
    print(f"Received query: {query!s}", file=sys.stderr)
    try:
        for token in rag.answer_query_stream(query):
            yield token
            full_response += token
    except Exception as e:
        print(f"Error in RAG system: {e}", file=sys.stderr)
        traceback.print_exc()
        yield "Internal Server Error"

    if not full_response:
        full_response = "No response generated"

    if analytics.write_key:
        # Track the query and response
        analytics.track(
            anonymous_id=anonymous_id,
            event="Chatbot Question submitted",
            properties={"query": query, "response": full_response, "source": source},
        )

    return full_response
