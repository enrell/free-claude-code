"""Tests for OpenCode Go provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers.base import ProviderConfig
from providers.opencode_go import OpenCodeGoProvider
from providers.opencode_go.request import OPENCODE_GO_DEFAULT_MAX_TOKENS


class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "opencode-go/glm-5"
        self.messages = [MockMessage("user", "Hello")]
        self.max_tokens = 100
        self.temperature = 0.5
        self.top_p = 0.9
        self.system = "System prompt"
        self.stop_sequences = None
        self.tools = []
        self.extra_body = {}
        self.thinking = None
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture
def opencode_go_config():
    return ProviderConfig(
        api_key="test_opencode_go_key",
        base_url="https://opencode.ai/zen/go/v1",
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    """Mock the global rate limiter to prevent waiting."""
    with patch("providers.openai_compat.GlobalRateLimiter") as mock:
        instance = mock.get_instance.return_value
        instance.wait_if_blocked = AsyncMock(return_value=False)

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        yield instance


@pytest.fixture
def opencode_go_provider(opencode_go_config):
    return OpenCodeGoProvider(opencode_go_config)


def test_init(opencode_go_config):
    """Test provider initialization."""
    with patch("providers.openai_compat.AsyncOpenAI") as mock_openai:
        provider = OpenCodeGoProvider(opencode_go_config)
        assert provider._api_key == "test_opencode_go_key"
        assert provider._base_url == "https://opencode.ai/zen/go/v1"
        mock_openai.assert_called_once()


def test_init_uses_configurable_timeouts():
    """Test that provider passes configurable read/write/connect timeouts to client."""
    config = ProviderConfig(
        api_key="test_opencode_go_key",
        base_url="https://opencode.ai/zen/go/v1",
        http_read_timeout=600.0,
        http_write_timeout=15.0,
        http_connect_timeout=5.0,
    )
    with patch("providers.openai_compat.AsyncOpenAI") as mock_openai:
        OpenCodeGoProvider(config)
        call_kwargs = mock_openai.call_args[1]
        timeout = call_kwargs["timeout"]
        assert timeout.read == 600.0
        assert timeout.write == 15.0
        assert timeout.connect == 5.0


def test_build_request_body_base_url_and_model(opencode_go_provider):
    """Base URL and model are correct in provider config."""
    assert opencode_go_provider._base_url == "https://opencode.ai/zen/go/v1"
    req = MockRequest(model="opencode-go/glm-5")
    body = opencode_go_provider._build_request_body(req)
    assert body["model"] == "opencode-go/glm-5"


def test_build_request_body_default_max_tokens(opencode_go_provider):
    """max_tokens=None uses OPENCODE_GO_DEFAULT_MAX_TOKENS (81920)."""
    req = MockRequest(max_tokens=None)
    body = opencode_go_provider._build_request_body(req)
    assert body["max_tokens"] == OPENCODE_GO_DEFAULT_MAX_TOKENS
    assert body["max_tokens"] == 81920


def test_build_request_body_with_thinking(opencode_go_provider):
    """Request body has extra_body.thinking.enabled when thinking is enabled."""
    thinking_mock = MagicMock()
    thinking_mock.enabled = True
    thinking_mock.effort = None
    req = MockRequest(thinking=thinking_mock)
    body = opencode_go_provider._build_request_body(req)

    assert body["model"] == "opencode-go/glm-5"
    assert body["temperature"] == 0.5
    assert len(body["messages"]) == 2
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == "System prompt"

    assert "extra_body" in body
    assert "thinking" in body["extra_body"]
    assert body["extra_body"]["thinking"]["enabled"] is True


def test_build_request_body_with_thinking_effort(opencode_go_provider):
    """Request body has extra_body.thinking with effort when thinking.effort is set."""
    from api.models.anthropic import ThinkingEffort

    thinking_mock = MagicMock()
    thinking_mock.enabled = True
    thinking_mock.effort = ThinkingEffort.HIGH
    req = MockRequest(thinking=thinking_mock)
    body = opencode_go_provider._build_request_body(req)

    assert "extra_body" in body
    assert "thinking" in body["extra_body"]
    assert body["extra_body"]["thinking"]["enabled"] is True
    assert body["extra_body"]["thinking"]["effort"] == "high"


def test_build_request_body_with_thinking_effort_max_mapped_to_high(
    opencode_go_provider,
):
    """OSS models don't support 'max' effort, so it gets mapped to 'high'."""
    from api.models.anthropic import ThinkingEffort

    thinking_mock = MagicMock()
    thinking_mock.enabled = True
    thinking_mock.effort = ThinkingEffort.MAX
    req = MockRequest(thinking=thinking_mock)
    body = opencode_go_provider._build_request_body(req)

    assert "extra_body" in body
    assert "thinking" in body["extra_body"]
    assert body["extra_body"]["thinking"]["enabled"] is True
    assert body["extra_body"]["thinking"]["effort"] == "high"


@pytest.mark.asyncio
async def test_stream_response_text(opencode_go_provider):
    """Test streaming text response."""
    import json

    req = MockRequest()

    mock_chunk1 = MagicMock()
    mock_chunk1.choices = [
        MagicMock(
            delta=MagicMock(content="Hello", reasoning_content=None),
            finish_reason=None,
        )
    ]
    mock_chunk1.usage = None

    mock_chunk2 = MagicMock()
    mock_chunk2.choices = [
        MagicMock(
            delta=MagicMock(content=" World", reasoning_content=None),
            finish_reason="stop",
        )
    ]
    mock_chunk2.usage = MagicMock(completion_tokens=10)

    async def mock_stream():
        yield mock_chunk1
        yield mock_chunk2

    with patch.object(
        opencode_go_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [e async for e in opencode_go_provider.stream_response(req)]

        assert len(events) > 0
        assert "event: message_start" in events[0]

        text_content = ""
        for e in events:
            if "event: content_block_delta" in e and '"text_delta"' in e:
                for line in e.splitlines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if "delta" in data and "text" in data["delta"]:
                            text_content += data["delta"]["text"]

        assert "Hello World" in text_content


@pytest.mark.asyncio
async def test_stream_response_error_path(opencode_go_provider):
    """Stream raises exception -> error event emitted."""
    req = MockRequest()

    async def mock_stream():
        raise RuntimeError("API failed")
        yield  # unreachable, makes it a generator

    with patch.object(
        opencode_go_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()
        events = [e async for e in opencode_go_provider.stream_response(req)]
        assert any("API failed" in e for e in events)
        assert any("message_stop" in e for e in events)
