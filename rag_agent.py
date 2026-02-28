"""
RAG Agent
---------
LangChain agent that chooses between:
  • vector_search   — semantic similarity (ChromaDB/FAISS)
  • keyword_search  — BM25 exact-match (part numbers, IDs, function names)
  • [MCP tools]     — live external sources (Confluence, JIRA, GitHub, filesystem)
                      loaded dynamically from mcp_servers.yaml

MCP integration strategy:
  The agent gets the full merged tool list. It uses local search tools for indexed
  PDFs and MCP tools for live/external data. The ReAct loop lets it combine both
  in a single response — e.g. find a spec in the PDF index, then look up the
  related JIRA ticket to check if it's been superseded.
"""
import logging
import os
import re
from typing import AsyncIterator, Optional

from langchain.agents import AgentExecutor, create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate

from app.config import Settings
from app.mcp.client_bridge import MCPClientBridge
from app.prompts.manager import PromptVersionManager
from app.tools.search_tools import KeywordSearchTool, VectorSearchTool

logger = logging.getLogger(__name__)

# Regex patterns for technical identifiers (heuristic for tool routing)
TECHNICAL_ID_PATTERN = re.compile(
    r'\b([A-Z]{2,}_[A-Z0-9_]+|0x[0-9A-Fa-f]+|[A-Z]{2,}\d{3,}|get_\w+\(|set_\w+\()\b'
)

MCP_CONFIG_PATH = os.getenv("MCP_CONFIG_PATH", "mcp_servers.yaml")


class RAGAgent:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.prompt_manager = PromptVersionManager(settings)
        self._embeddings: Optional[OpenAIEmbeddings] = None
        self._executor_cache: dict[str, AgentExecutor] = {}
        self._mcp_bridge = MCPClientBridge(config_path=MCP_CONFIG_PATH)
        self._mcp_tools: list | None = None   # loaded once, then cached

    async def initialize(self):
        """Called once at startup to warm up embeddings and MCP connections."""
        self._embeddings = OpenAIEmbeddings(model=self.settings.embedding_model)
        logger.info("Embedding model loaded.")

        # Pre-load MCP tools so first query isn't slow
        self._mcp_tools = await self._mcp_bridge.get_tools()
        if self._mcp_tools:
            names = [t.name for t in self._mcp_tools]
            logger.info(f"MCP tools loaded: {names}")
        else:
            logger.info("No MCP tools configured (mcp_servers.yaml not found or all disabled).")

    async def is_healthy(self) -> bool:
        """Health check: verify vector store is accessible."""
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

        # Heuristic pre-routing hint (agent can override)
        has_technical_id = bool(TECHNICAL_ID_PATTERN.search(question))
        routing_hint = "keyword" if has_technical_id else "semantic"
        logger.info(f"Query routing hint: {routing_hint} | question: {question[:80]}")

        result = await executor.ainvoke({"input": question})

        # Determine which tool(s) were actually used
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
        """Streaming version for the /query/stream endpoint."""
        executor = self._get_executor(prompt_version)
        async for chunk in executor.astream({"input": question}):
            if "output" in chunk:
                yield chunk["output"]

    def _get_executor(
        self, prompt_version: str, filters: Optional[dict] = None
    ) -> AgentExecutor:
        """
        Build (or retrieve cached) AgentExecutor for a given prompt version.
        Tool list = local RAG tools + any enabled MCP client tools.
        """
        cache_key = f"{prompt_version}:{str(filters)}"
        if cache_key in self._executor_cache:
            return self._executor_cache[cache_key]

        prompt_data = self.prompt_manager.load(prompt_version)
        llm = self._build_llm()

        # ── Core local tools ──────────────────────────────────
        local_tools = [
            VectorSearchTool(settings=self.settings, embeddings=self._embeddings, filters=filters),
            KeywordSearchTool(settings=self.settings),
        ]

        # ── MCP client tools (external live sources) ──────────
        # These are pre-loaded at startup. If none configured, list is empty.
        mcp_tools = self._mcp_tools or []

        tools = local_tools + mcp_tools
        tool_guidance = self._build_tool_guidance(mcp_tools)

        # ReAct prompt: Thought → Action → Observation loop
        react_template = (
            f"{prompt_data['system']}\n\n"
            "You have access to the following tools:\n\n"
            "{tools}\n\n"
            f"{tool_guidance}\n\n"
            "Use this format:\n"
            "Thought: reason about which tool to use\n"
            "Action: the tool name\n"
            "Action Input: the input to the tool\n"
            "Observation: the tool result\n"
            "... (repeat if needed)\n"
            "Thought: I have enough information to answer\n"
            "Final Answer: your complete answer with source citations\n\n"
            "Tool names: {tool_names}\n\n"
            "{agent_scratchpad}\n\n"
            f"{prompt_data['agent_prefix']}"
        )
        prompt = PromptTemplate.from_template(react_template)

        agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=6,   # Increased: agent may need extra steps with MCP tools
            handle_parsing_errors=True,
        )

        self._executor_cache[cache_key] = executor
        return executor

    def _build_tool_guidance(self, mcp_tools: list) -> str:
        """Generate dynamic tool routing guidance based on what's available."""
        base = (
            "Tool selection guide:\n"
            "  • vector_search   → conceptual questions, 'how does X work'\n"
            "  • keyword_search  → part numbers, error codes, function names, exact IDs\n"
        )
        if not mcp_tools:
            return base

        mcp_names = [t.name for t in mcp_tools]
        mcp_guidance = "\n".join(f"  • {name}" for name in mcp_names)
        return (
            base
            + "  • MCP tools below → live external data (use AFTER local search if needed):\n"
            + mcp_guidance
            + "\n\nStrategy: Always search local indexed PDFs first. "
            "Use MCP tools to supplement with live data (tickets, wikis, source code)."
        )

    def _build_llm(self):
        if self.settings.llm_provider == "anthropic":
            return ChatAnthropic(
                model=self.settings.llm_model,
                api_key=self.settings.anthropic_api_key,
                streaming=True,
            )
        return ChatOpenAI(
            model=self.settings.llm_model,
            api_key=self.settings.openai_api_key,
            streaming=True,
        )

    def _determine_strategy(self, tool_calls: list[str]) -> str:
        used_vector = "vector_search" in tool_calls
        used_keyword = "keyword_search" in tool_calls
        used_mcp = any(t for t in tool_calls if "__" in t)  # MCP tools use "server__tool" naming

        if used_mcp and (used_vector or used_keyword):
            return "hybrid+mcp"
        if used_mcp:
            return "mcp"
        if used_vector and used_keyword:
            return "hybrid"
        if used_keyword:
            return "keyword"
        return "semantic"

    def _extract_sources(self, intermediate_steps: list) -> list[dict]:
        """Parse source metadata from agent's intermediate steps."""
        sources = []
        seen = set()
        for action, observation in intermediate_steps:
            if isinstance(observation, str):
                # Extract filename/page from formatted tool output
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
