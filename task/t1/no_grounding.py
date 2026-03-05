import asyncio
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

# Note:
# Before implementation open the `flow_diagram.png` to see the flow of app

BATCH_SYSTEM_PROMPT = """You are a user search assistant. Your task is to find users from the provided list that match the search criteria.

INSTRUCTIONS:
1. Analyze the user question to understand what attributes/characteristics are being searched for
2. Examine each user in the context and determine if they match the search criteria
3. For matching users, extract and return their complete information
4. Be inclusive - if a user partially matches or could potentially match, include them

OUTPUT FORMAT:
- If you find matching users: Return their full details exactly as provided, maintaining the original format
- If no users match: Respond with exactly "NO_MATCHES_FOUND"
- If uncertain about a match: Include the user with a note about why they might match"""

FINAL_SYSTEM_PROMPT = """You are a helpful assistant that provides comprehensive answers based on user search results.

INSTRUCTIONS:
1. Review all the search results from different user batches
2. Combine and deduplicate any matching users found across batches
3. Present the information in a clear, organized manner
4. If multiple users match, group them logically
5. If no users match, explain what was searched for and suggest alternatives"""

USER_PROMPT = """## USER DATA:
{context}

## SEARCH QUERY: 
{query}"""


class TokenTracker:
    def __init__(self):
        self.total_tokens = 0
        self.batch_tokens = []

    def add_tokens(self, tokens: int):
        self.total_tokens += tokens
        self.batch_tokens.append(tokens)

    def get_summary(self):
        return {
            'total_tokens': self.total_tokens,
            'batch_count': len(self.batch_tokens),
            'batch_tokens': self.batch_tokens
        }


llm_client = AzureChatOpenAI(
    # Basic, deterministic configuration suitable for experiments.
    temperature=0.0,
    azure_deployment="gpt-4o",
    azure_endpoint=DIAL_URL,
    api_key=SecretStr(API_KEY),
    # Adjust this if your DIAL setup expects a different version.
    api_version="2024-02-15-preview",
)

token_tracker = TokenTracker()


def join_context(context: list[dict[str, Any]]) -> str:
    context_str = ""
    for user in context:
        context_str += "User:\n"
        for key, value in user.items():
            context_str += f"  {key}: {value}\n"
        context_str += "\n"
    return context_str


async def generate_response(system_prompt: str, user_message: str) -> str:
    print("Processing...")
    # 1. Create messages list for the chat model.
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    # 2. Call the model asynchronously.
    response = await llm_client.ainvoke(messages)

    # 3. Read token usage information if it is available.
    token_usage = response.response_metadata.get("token_usage", {}) if hasattr(response, "response_metadata") else {}
    total_tokens = token_usage.get("total_tokens", 0)

    # 4. Track token usage for later summary.
    if isinstance(total_tokens, int):
        token_tracker.add_tokens(total_tokens)

    # 5. Show the model answer and token usage.
    print("\nModel response:")
    print(response.content)
    print(f"\nTokens used in this call: {total_tokens}")

    # 6. Return raw content so the caller can work with it.
    return response.content


async def main():
    print("Query samples:")
    print(" - Do we have someone with name John that loves traveling?")

    # Run blocking input() in a worker thread so it does not block the event loop.
    user_question = (await asyncio.to_thread(input, "> ")).strip()
    if user_question:
        print("\n--- Searching user database ---")

        # 1. Get all users.
        user_client = UserClient()
        users = user_client.get_all_users()

        # 2. Split users into batches of at most 100 users.
        batch_size = 100
        user_batches: list[list[dict[str, Any]]] = [
            users[i : i + batch_size] for i in range(0, len(users), batch_size)
        ]

        # 3. Prepare asynchronous tasks: one LLM call per batch.
        tasks: list[asyncio.Future[str]] = []
        for user_batch in user_batches:
            batch_context = join_context(user_batch)
            user_message = USER_PROMPT.format(context=batch_context, query=user_question)
            tasks.append(generate_response(system_prompt=BATCH_SYSTEM_PROMPT, user_message=user_message))

        # 4. Run all batch searches in parallel.
        batch_results = await asyncio.gather(*tasks)

        # 5. Filter out batches that reported no matches.
        relevant_results = [
            result for result in batch_results if result.strip() != "NO_MATCHES_FOUND"
        ]

        # 6. If we have any matches, combine them and ask the model for a final answer.
        if relevant_results:
            combined_results = "\n\n".join(relevant_results)
            final_message = (
                f"SEARCH RESULTS:\n{combined_results}\n\nORIGINAL QUERY: {user_question}"
            )
            await generate_response(
                system_prompt=FINAL_SYSTEM_PROMPT,
                user_message=final_message,
            )
        else:
            print("No users found matching the given search query.")

        # 7. Show a short summary of token usage.
        usage_summary = token_tracker.get_summary()
        print("\n--- Token usage summary ---")
        print(f"Total tokens: {usage_summary['total_tokens']}")
        print(f"Number of batch calls: {usage_summary['batch_count']}")
        print(f"Tokens per batch: {usage_summary['batch_tokens']}")


if __name__ == "__main__":
    asyncio.run(main())


# The problems with No Grounding approach are:
#   - If we load whole users as context in one request to LLM we will hit context window
#   - Huge token usage == Higher price per request
#   - Added + one chain in flow where original user data can be changed by LLM (before final generation)
# User Question -> Get all users -> ‼️parallel search of possible candidates‼️ -> probably changed original context -> final generation