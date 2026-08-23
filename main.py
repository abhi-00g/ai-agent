"""
Main entry point for ATLAS — Multi-Tool AI Agent.

Run this to start an interactive chat session:
    python main.py

Type your questions, see ATLAS think step by step, and type 'quit' to exit.
Type 'reset' to clear conversation history and start fresh.
"""

from agent import Agent


def main():
    print("=" * 60)
    print("  ATLAS — Multi-Tool AI Agent")
    print('  "I carry the weight so you don\'t have to."')
    print()
    print("  Type 'quit' to exit, 'reset' to clear history")
    print("=" * 60)
    print()

    agent = Agent()

    # Show which tools are available
    tools = agent.registry.list_tools()
    print(f"Tools: {', '.join(tools)}")

    # Show active guardrails
    blocked = agent.guardrails.list_blocked_topics()
    if blocked:
        print(f"Safety guardrails active: {len(blocked)} blocked topics")

    print()

    while True:
        # Get user input
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nATLAS: Until next time. Stay curious!")
            break

        # Handle commands
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("ATLAS: Until next time. Stay curious!")
            break
        if user_input.lower() == "reset":
            agent.reset()
            print("Conversation history cleared.\n")
            continue

        # Send to agent and get response
        print()
        response = agent.chat(user_input)
        print(f"\nATLAS: {response}\n")


if __name__ == "__main__":
    main()
