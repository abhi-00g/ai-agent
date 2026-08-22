"""
Main entry point for the AI Agent.

Run this to start an interactive chat session:
    python main.py

Type your questions, see the agent think step by step, and type 'quit' to exit.
Type 'reset' to clear conversation history and start fresh.
"""

from agent import Agent


def main():
    print("=" * 60)
    print("  AI Agent — Multi-Tool Assistant")
    print("  Type 'quit' to exit, 'reset' to clear history")
    print("=" * 60)
    print()

    agent = Agent()

    # Show which tools are available
    tools = agent.registry.list_tools()
    print(f"Available tools: {', '.join(tools)}")
    print()

    while True:
        # Get user input
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        # Handle commands
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            agent.reset()
            print("Conversation history cleared.\n")
            continue

        # Send to agent and get response
        print()  # Blank line before agent's thinking
        response = agent.chat(user_input)
        print(f"\nAgent: {response}\n")


if __name__ == "__main__":
    main()
