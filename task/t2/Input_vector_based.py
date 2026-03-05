import asyncio
from typing import Any
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

# Note:
# Before implementation open the `vector_based_grounding.png` to see the flow of app


SYSTEM_PROMPT = """You are a RAG-powered assistant that assists users with their questions about user information.
            
## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
"""

USER_PROMPT = """##RAG CONTEXT:
{context}

##USER QUESTION: 
{query}"""


def format_user_document(user: dict[str, Any]) -> str:
    """
    Convert a single user dictionary into a readable text representation.

    This mirrors the formatting used in `join_context` from `no_grounding.py`.
    """
    lines: list[str] = ["User:"]
    for key, value in user.items():
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)


class UserRAG:
    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = None

    async def __aenter__(self):
        print("🔎 Loading all users...")
        # 1. Get all users from the service.
        user_client = UserClient()
        users = user_client.get_all_users()

        # 2. Build one Document per user.
        documents: list[Document] = [
            Document(page_content=format_user_document(user)) for user in users
        ]

        print(f"↗️ Creating embeddings and vectorstore for {len(documents)} documents...")
        # Create a FAISS vector store in batches to respect context limits.
        self.vectorstore = await self._create_vectorstore_with_batching(documents)
        print("✅ Vectorstore is ready.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # No async cleanup is required for the vectorstore at the moment.
        # The method exists to keep the async context manager interface explicit.
        return False

    async def _create_vectorstore_with_batching(self, documents: list[Document], batch_size: int = 100):
        """
        Create a single FAISS vectorstore from documents by processing them in batches.

        Batching helps to stay within embedding model context limits.
        """
        if not documents:
            raise ValueError("No documents provided to build the vectorstore.")

        # 1. Split documents into batches.
        batches: list[list[Document]] = [
            documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
        ]

        # 2. Create async tasks for building partial FAISS indexes.
        batch_tasks = [FAISS.afrom_documents(batch, self.embeddings) for batch in batches]

        # 3. Run all embedding tasks concurrently.
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        # 4. Merge all partial vectorstores into one.
        final_vectorstore = None
        for result in batch_results:
            if isinstance(result, Exception):
                # In a real system you might want better error handling.
                print(f"Error while creating batch vectorstore: {result}")
                continue

            batch_vectorstore = result
            if batch_vectorstore is None:
                continue

            if final_vectorstore is None:
                final_vectorstore = batch_vectorstore
            else:
                final_vectorstore.merge_from(batch_vectorstore)

        # 5. Validate that we have a final vectorstore and return it.
        if final_vectorstore is None:
            raise RuntimeError("Vectorstore creation failed for all batches.")

        return final_vectorstore

    async def retrieve_context(self, query: str, k: int = 10, score: float = 0.1) -> str:
        """
        Retrieve a textual context from the vectorstore for a given query.
        """
        if self.vectorstore is None:
            raise RuntimeError("Vectorstore is not initialized.")

        # 1. Perform similarity search with relevance scores in a worker thread
        #    so that we do not block the event loop.
        results = await asyncio.to_thread(
            self.vectorstore.similarity_search_with_relevance_scores,
            query,
            k=k,
            score_threshold=score,
        )

        # 2. Collect page contents from the retrieved documents.
        context_parts: list[str] = []
        for doc, relevance_score in results:
            context_parts.append(doc.page_content)
            print(f"Retrieved (Score: {relevance_score:.3f}): {doc.page_content}")

        # 3. Join context parts with extra newlines for readability.
        return "\n\n".join(context_parts)

    def augment_prompt(self, query: str, context: str) -> str:
        """Combine user query and retrieved context into a single prompt."""
        return USER_PROMPT.format(context=context, query=query)

    def generate_answer(self, augmented_prompt: str) -> str:
        """
        Call the chat model with the RAG-style prompt and return its answer.
        """
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=augmented_prompt),
        ]
        response = self.llm_client.invoke(messages)
        return response.content


async def main():
    embeddings = AzureOpenAIEmbeddings(
        # Embedding model configuration.
        deployment="text-embedding-3-small-1",
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        dimensions=384,
    )

    llm_client = AzureChatOpenAI(
        temperature=0.0,
        azure_deployment="gpt-4o",
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        api_version="2024-02-15-preview",
    )

    async with UserRAG(embeddings, llm_client) as rag:
        print("Query samples:")
        print(" - I need user emails that filled with hiking and psychology")
        print(" - Who is John?")
        while True:
            # Run blocking input in a worker thread so it does not block the event loop.
            user_question = (await asyncio.to_thread(input, "> ")).strip()
            if user_question.lower() in ['quit', 'exit']:
                break
            # 1. Retrieve relevant context from the vectorstore.
            context = await rag.retrieve_context(user_question)
            if not context:
                print("No relevant information found in vector store.")
                continue

            # 2. Build an augmented prompt that combines context and question.
            augmented_prompt = rag.augment_prompt(user_question, context)

            # 3. Generate and print the answer.
            answer = rag.generate_answer(augmented_prompt)
            print("\nAnswer:")
            print(answer)


asyncio.run(main())

# The problems with Vector based Grounding approach are:
#   - In current solution we fetched all users once, prepared Vector store (Embed takes money) but we didn't play
#     around the point that new users added and deleted every 5 minutes. (Actually, it can be fixed, we can create once
#     Vector store and with new request we will fetch all the users, compare new and deleted with version in Vector
#     store and delete the data about deleted users and add new users).
#   - Limit with top_k (we can set up to 100, but what if the real number of similarity search 100+?)
#   - With some requests works not so perfectly. (Here we can play and add extra chain with LLM that will refactor the
#     user question in a way that will help for Vector search, but it is also not okay in the point that we have
#     changed original user question).
#   - Need to play with balance between top_k and score_threshold
# Benefits are:
#   - Similarity search by context
#   - Any input can be used for search
#   - Costs reduce