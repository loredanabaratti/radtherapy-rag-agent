"""
Run an interactive demo of the RadTherapy RAG Assistant.
"""

from src.graph import build_graph

def print_result(result: dict) -> None:
    """Prints the result in a readable format."""

    answer = result.get("answer", "No answer found.")

    print("\n" + "="*70)
    print("ANSWER")
    print("="*70)
    print(answer["answer"])

    print("\nSOURCES")
    print("="*70)

    if not answer.get("source_indices"):
        print("No sources found.")
    else:
        seen = set()
        for index in answer["source_indices"]:
            source = result["retrieved_sources"][index]["reference"]
            key = (source["title"], source["url"])
            if key in seen:
                continue
            seen.add(key)
            print(f"Title: {source['title']}")
            print(f"Source: {source['source']}")
            print(f"URL: {source['url']}")

    critic = result.get("critic_result", {})

    print("\nGROUNDING CHECK")
    print("="*70)
    print(f"Grounded: {critic.get('is_grounded', 'Unknown')}")
    print(f"Feedback: {critic.get('feedback', 'No feedback provided.')}")
    print(f"Retry count: {result.get('retry_count', 'Unknown')}")
    print("="*70)

def main() -> None:
    """Main function to run the demo."""

    graph = build_graph()

    print("RadTherapy RAG Assistant Demo")
    print("Type your question below (or type 'exit' to quit):")

    while True:
        user_input = input("\nYour question: ").strip()
        if user_input.lower() == "exit":
            print("Exiting the demo. Goodbye!")
            break

        if not user_input:
            print("Please enter a valid question.")
            continue

        initial_state = {
            "question": user_input,
            "router_decision": None,
            "retrieved_sources": [],
            "answer": None,
            "critic_result": None,
            "retry_count": 0,
        }
        result = graph.invoke(initial_state)
        print_result(result)

if __name__ == "__main__":
    main()