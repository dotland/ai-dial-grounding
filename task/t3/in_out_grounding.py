import asyncio
from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr, BaseModel, Field
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

# Note: HOBBIES SEARCHING WIZARD
# - Before implementation open the `flow.png` to see the flow of the app.
# - Searches users by hobbies and provides their full info in JSON format.
# - Only `id` and `about_me` are embedded to keep the context window small.
# - The vector store is incrementally updated so it stays in sync with the user service.

SYSTEM_PROMPT = """You are a RAG-powered assistant that groups users by their hobbies.

## Flow:
Step 1: User will ask to search users by their hobbies etc.
Step 2: Will be performed search in the Vector store to find most relevant users.
Step 3: You will be provided with CONTEXT (most relevant users, there will be user ID and information about user), and 
        with USER QUESTION.
Step 4: You group by hobby users that have such hobby and return response according to Response Format

## Response Format:
{format_instructions}
"""

USER_PROMPT = """## CONTEXT:
{context}

## USER QUESTION: 
{query}"""


llm_client = AzureChatOpenAI(
    temperature=0.0,
    azure_deployment="gpt-4o",
    azure_endpoint=DIAL_URL,
    api_key=SecretStr(API_KEY),
    api_version="2024-02-15-preview",
)


class GroupingResult(BaseModel):
    hobby: str = Field(description="Hobby. Example: football, painting, horsing, photography, bird watching...")
    user_ids: list[int] = Field(description="List of user IDs that have hobby requested by user.")


class GroupingResults(BaseModel):
    grouping_results: list[GroupingResult] = Field(description="List matching search results.")


def format_user_document(user: dict[str, Any]) -> str:
    """
    Build a compact text representation of a user that includes only
    their id and about_me section, suitable for hobby-based search.
    """
    return "\n".join(
        [
            "User:",
            f"  id: {user.get('id')}",
            f"  About user: {user.get('about_me')}",
        ]
    )


class InputGrounder:
    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.user_client = UserClient()
        self.vectorstore = None

    async def __aenter__(self):
        await self.initialize_vectorstore()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # No async cleanup is required at the moment.
        # Returning False lets any exception propagate normally.
        return False

    async def initialize_vectorstore(self, batch_size: int = 50):
        """Initialize vectorstore with all current users."""
        print("🔍 Loading all users for initial vectorstore...")
        # 1. Get all users (use UserClient).
        users = self.user_client.get_all_users()

        # 2. Prepare Documents with id and about_me only.
        documents: list[Document] = [
            Document(id=str(user.get("id")), page_content=format_user_document(user))
            for user in users
        ]

        if not documents:
            print("No users found to initialize the vectorstore.")
            return

        # 3. Split documents into batches so we do not exceed embedding context limits.
        batches: list[list[Document]] = [
            documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
        ]

        # 4. Create the Chroma vectorstore and add all document batches.
        self.vectorstore = Chroma(
            collection_name="users",
            embedding_function=self.embeddings,
        )

        tasks = [self.vectorstore.aadd_documents(batch) for batch in batches]
        await asyncio.gather(*tasks)

    async def retrieve_context(self, query: str, k: int = 100, score: float = 0.2) -> str:
        """Retrieve context, with optional automatic vectorstore update."""
        if self.vectorstore is None:
            raise RuntimeError("Vectorstore is not initialized.")

        # 1. Keep vectorstore in sync with the latest users.
        await self._update_vectorstore()

        # 2. Run similarity search with relevance scores.
        results = await asyncio.to_thread(
            self.vectorstore.similarity_search_with_relevance_scores,
            query,
            k=k,
            score_threshold=score,
        )

        # 3. Collect relevant document contents.
        context_parts: list[str] = []
        for doc, relevance_score in results:
            context_parts.append(doc.page_content)
            print(f"Retrieved (Score: {relevance_score:.3f}): {doc.page_content}")

        # 4. Join everything into a single context string.
        return "\n\n".join(context_parts)

    async def _update_vectorstore(self):
        if self.vectorstore is None:
            return

        # 1. Get all users from the service.
        users = self.user_client.get_all_users()

        # 2. Get all ids currently stored in the vectorstore.
        vectorstore_data = self.vectorstore.get()
        vectorstore_ids_set = {
            str(user_id) for user_id in vectorstore_data.get("ids", [])
        }

        # 3. Prepare a dictionary of users keyed by id.
        users_dict: dict[str, dict[str, Any]] = {
            str(user.get("id")): user for user in users
        }
        users_ids_set = set(users_dict.keys())

        # 4. Determine new and deleted user ids.
        ids_to_add = users_ids_set - vectorstore_ids_set
        ids_to_delete = vectorstore_ids_set - users_ids_set

        # 5. Remove deleted users from the vectorstore.
        if ids_to_delete:
            self.vectorstore.delete(ids=list(ids_to_delete))

        # 6. Add new users to the vectorstore.
        new_documents: list[Document] = [
            Document(id=user_id, page_content=format_user_document(users_dict[user_id]))
            for user_id in ids_to_add
        ]

        if new_documents:
            await self.vectorstore.aadd_documents(new_documents)

    def augment_prompt(self, query: str, context: str) -> str:
        """Combine user query and retrieved context into a single prompt."""
        return USER_PROMPT.format(context=context, query=query)

    def generate_answer(self, augmented_prompt: str) -> GroupingResults:
        # 1. Parser for the structured grouping result.
        parser = PydanticOutputParser(pydantic_object=GroupingResults)

        # 2. System prompt template and human message.
        system_prompt = SystemMessagePromptTemplate.from_template(template=SYSTEM_PROMPT)
        messages = [
            system_prompt,
            HumanMessage(content=augmented_prompt),
        ]

        # 3. Build chat prompt with format instructions filled in.
        prompt = ChatPromptTemplate.from_messages(messages=messages).partial(
            format_instructions=parser.get_format_instructions()
        )

        # 4. Run the LCEL chain and parse the result.
        grouping_results: GroupingResults = (prompt | llm_client | parser).invoke({})
        return grouping_results


class OutputGrounder:
    def __init__(self):
        self.user_client = UserClient()

    async def ground_response(self, grouping_results: GroupingResults):
        """
        Perform output grounding: fetch full user data for the IDs suggested by the model.
        """
        for grouping_result in grouping_results.grouping_results:
            print(f"\nHobby: {grouping_result.hobby}")
            users = await self._find_users(grouping_result.user_ids)
            if users:
                print("Users:")
                for user in users:
                    print(user)
            else:
                print("No valid users found for this hobby.")

    async def _find_users(self, ids: list[int]) -> list[dict[str, Any]]:
        async def safe_get_user(user_id: int) -> Optional[dict[str, Any]]:
            try:
                # Get user by id (it is async method).
                return await self.user_client.get_user(user_id)
            except Exception as e:
                if "404" in str(e):
                    print(f"User with ID {user_id} is absent (404)")
                    return None
                raise  # Re-raise non-404 errors

        # 1. Prepare tasks to fetch users concurrently.
        tasks = [safe_get_user(user_id) for user_id in ids]
        results = await asyncio.gather(*tasks)

        # 2. Filter out missing users.
        return [user for user in results if user is not None]


async def main():
    embeddings = AzureOpenAIEmbeddings(
        deployment="text-embedding-3-small-1",
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        dimensions=384,
        check_embedding_ctx_length=False,
    )
    output_grounder = OutputGrounder()

    async with InputGrounder(embeddings, llm_client) as rag:
        print("Query samples:")
        print(" - I need people who love to go to mountains")
        print(" - Find people who love to watch stars and night sky")
        print(" - I need people to go to fishing together")

        while True:
            # Run blocking input() in a worker thread so the event loop stays responsive.
            user_question = (await asyncio.to_thread(input, "> ")).strip()
            if user_question.lower() in ['quit', 'exit']:
                break
            # 1. Retrieve context for the query.
            context = await rag.retrieve_context(user_question)
            if not context:
                print("No relevant information found in vectorstore.")
                continue

            # 2. Build augmented prompt.
            augmented_prompt = rag.augment_prompt(user_question, context)

            # 3. Generate structured grouping results.
            grouping_results = rag.generate_answer(augmented_prompt)

            # 4. Ground the output using the live user service.
            await output_grounder.ground_response(grouping_results)


if __name__ == "__main__":
    asyncio.run(main())
