from typing import Any, List, Optional, Dict
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import Field
import cohere

class CohereV2ChatModel(BaseChatModel):
    """Custom LangChain chat model wrapper for Cohere V2 API."""
    model_name: str = Field(alias="model")
    api_key: str
    temperature: float = 0.1
    max_tokens: int = 4096

    @property
    def _llm_type(self) -> str:
        return "cohere-v2-chat"

    def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        cohere_msgs = []
        for m in messages:
            if isinstance(m, HumanMessage):
                cohere_msgs.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                cohere_msgs.append({"role": "assistant", "content": m.content})
            elif isinstance(m, SystemMessage):
                cohere_msgs.append({"role": "system", "content": m.content})
            else:
                cohere_msgs.append({"role": "user", "content": m.content})
        return cohere_msgs

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        client = cohere.ClientV2(api_key=self.api_key)
        res = client.chat(
            model=self.model_name,
            messages=self._convert_messages(messages),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop_sequences=stop,
            **kwargs
        )
        msg = AIMessage(content=res.message.content[0].text)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        client = cohere.AsyncClientV2(api_key=self.api_key)
        res = await client.chat(
            model=self.model_name,
            messages=self._convert_messages(messages),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop_sequences=stop,
            **kwargs
        )
        msg = AIMessage(content=res.message.content[0].text)
        return ChatResult(generations=[ChatGeneration(message=msg)])
