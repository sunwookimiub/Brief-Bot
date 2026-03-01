"""
RAG Agent
---------
LangChain agent that chooses between vector_search and keyword_search
based on the query. Logs prompt version for every response.
"""
import logging
import re
from typing import AsyncIterator, Optional

from langchain.agents import AgentExecutor, create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_voyageai import VoyageAIEmbeddings
from langchain.prompts import PromptTemplate

from app.config import Settings
from app.prompts.manager import PromptVersionManager
from app.tools.search_tools import KeywordSearchTool, VectorSearchTool

logger = logging.getLogger(__name__)

TECHNICAL_ID_PATTERN = re.compile(
    r'\b([A-Z]{2,}_[A-Z0-9_]+|0x[0-9A-Fa-f]+|[A-Z]{2,}\d{3,}|get_\w+\(|set_\w+\()\b'
)


class RAGAgent:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.prompt_manager = PromptVersionManager(settings)
        self._embeddings: Optional[VoyageAIEmbeddings] = None
        self._executor_cache: dict[str, AgentExecutor] = {}

    async def initialize(self):
        self._embeddings = VoyageAIEmbeddings(
            voyage_api_key=self.settings.voyage_api_key,
            model=self.settings.embedding_model,
        )
        logger.info("Embedding model loaded.")

    async def is_healthy(self) -> bool:
        try:
            from langchain_community.vectorstores import Chroma
            Chroma(
                persist_directory=self.settings.chroma_persist_dir,
                embedding_function=self._embeddings,
                collection_name="rag_docs",
            )
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def query(
        self,
        question: str,
        prompt_version: str = "v1",
        filters: Optional[dict] = None,
    ) -> dict:
        executor = self._get_executor(prompt_version, filters)

        has_technical_id = bool(TECHNICAL_ID_PATTERN.search(question))
        routing_hint = "keyword" if has_technical_id else "semantic"
        logger.info(f"Query routing hint: {routing_hint} | question: {question[:80]}")

        result = await executor.ainvoke({"input": question})

        tool_calls = [step[0].tool for step in result.get("intermediate_steps", [])]
        strategy = self._determine_strategy(tool_calls)

        return {
            "answer": result["output"],
            "sources": self._extract_sources(result.get("intermediate_steps", [])),
            "search_strategy": strategy,
            "prompt_version": prompt_version,
        }

    async def query_stream(
        self, question: str, prompt_version: str = "v1"
    ) -> AsyncIterator[str]:
        executor = self._get_executor(prompt_version)
        async for chunk in executor.astream({"input": question}):
            if "output" in chunk:
                yield chunk["output"]

    def _get_executor(
        self, prompt_version: str, filters: Optional[dict] = None
    ) -> AgentExecutor:
        cache_key = f"{prompt_version}:{str(filters)}"
        if cache_key in self._executor_cache:
            return self._executor_cache[cache_key]

        prompt_data = self.prompt_manager.load(prompt_version)
        llm = self._build_llm()

        tools = [
            VectorSearchTool(settings=self.settings, embeddings=self._embeddings, filters=filters),
            KeywordSearchTool(settings=self.settings),
        ]

        # Build prompt using explicit input_variables to avoid Python string
        # format() consuming {input} before LangChain can use it
        template_str = (
            prompt_data["system"] + "\n\n"
            "You have access to the following tools:\n\n"
            "{tools}\n\n"
            "Tool selection guide:\n"
            "  - vector_search: conceptual questions, how things work\n"
            "  - keyword_search: exact part numbers, error codes, function names\n\n"
            "Use EXACTLY this format:\n"
            "Thought: reason about which tool to use\n"
            "Action: tool name (must be one of [{tool_names}])\n"
            "Action Input: your search query\n"
            "Observation: the tool result\n"
            "... (repeat as needed)\n"
            "Thought: I now have enough information\n"
            "Final Answer: complete answer with filename and page citations\n\n"
            "Begin!\n\n"
            "Question: {input}\n"
            "{agent_scratchpad}"
        )

        prompt = PromptTemplate(
            template=template_str,
            input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
        )

        agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=4,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

        self._executor_cache[cache_key] = executor
        return executor

    def _build_llm(self):
        return ChatAnthropic(
            model=self.settings.llm_model,
            api_key=self.settings.anthropic_api_key,
            streaming=True,
        )

    def _determine_strategy(self, tool_calls: list[str]) -> str:
        used_vector = "vector_search" in tool_calls
        used_keyword = "keyword_search" in tool_calls
        if used_vector and used_keyword:
            return "hybrid"
        if used_keyword:
            return "keyword"
        return "semantic"

    def _extract_sources(self, intermediate_steps: list) -> list[dict]:
        sources = []
        seen = set()
        for action, observation in intermediate_steps:
            if isinstance(observation, str):
                for line in observation.split("\n"):
                    if "File:" in line and "Page:" in line:
                        key = line.strip()
                        if key not in seen:
                            seen.add(key)
                            sources.append({
                                "tool": action.tool,
                                "reference": key,
                            })
        return sources
