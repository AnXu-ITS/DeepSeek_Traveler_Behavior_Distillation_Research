"""Teacher clients: real DeepSeek-compatible client + deterministic mock."""
from __future__ import annotations

import json
import os
import time

import httpx

from ..schemas.state import UniversalTravelerState
from .prompts import SYSTEM_PROMPT, build_user_prompt


class TeacherClientError(Exception):
    def __init__(self, failure_type: str, message: str):
        self.failure_type = failure_type
        super().__init__(message)


class DeepSeekTeacherClient:
    """OpenAI-compatible chat completions client.

    Endpoint / key / model are read from constructor args, then ``DEEPSEEK_*``
    environment variables, then sensible defaults. Never hard-coded.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_retries: int = 2,
        timeout_seconds: float = 120.0,
        max_tokens: int = 8192,
        cache_bypass: bool = False,
    ):
        # Environment variables take precedence over constructor args so the
        # endpoint/key/model can always be configured externally.
        self.base_url = (os.environ.get("DEEPSEEK_BASE_URL") or base_url or "").rstrip("/")
        self.api_key = os.environ.get("DEEPSEEK_API_KEY") or api_key
        self.model = os.environ.get("DEEPSEEK_MODEL") or model or "deepseek-v4-pro"
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout_seconds
        self.max_tokens = max_tokens
        # When True, adds LiteLLM's ``cache: {"no-cache": True}`` request field
        # (audit mode only). This disables the gateway response cache WITHOUT
        # touching the behavioral prompt. Production calls leave this False.
        self.cache_bypass = cache_bypass

        if not self.base_url:
            raise TeacherClientError("config_error", "no base_url provided (DEEPSEEK_BASE_URL)")
        if not self.api_key:
            raise TeacherClientError("config_error", "no api_key provided (DEEPSEEK_API_KEY)")

    def build_prompt(self, state: UniversalTravelerState) -> tuple[str, str]:
        """Return ``(system_prompt, user_prompt)`` exactly as sent to the model."""
        return SYSTEM_PROMPT, build_user_prompt(state.model_dump_json(indent=2))

    def respond(self, state: UniversalTravelerState, cache_bypass: bool | None = None) -> str:
        """Return raw teacher response text (JSON), retrying on failure."""
        return self.respond_with_evidence(state, cache_bypass=cache_bypass)["content"]

    def respond_with_evidence(
        self, state: UniversalTravelerState, cache_bypass: bool | None = None
    ) -> dict:
        """Like :meth:`respond` but also returns request/response evidence.

        Returns ``{"content": str, "evidence": dict}``. ``evidence`` contains
        only non-secret metadata (never the API key or Authorization header):
        HTTP status, elapsed seconds, model, finish_reason, completion id,
        server ``created`` timestamp, usage, and a curated set of safe response
        headers (including gateway cache indicators when present).

        ``cache_bypass=True`` adds LiteLLM's ``cache: {"no-cache": True}`` to
        the request body (gateway metadata only, not part of the prompt).
        """
        use_bypass = self.cache_bypass if cache_bypass is None else cache_bypass
        system_prompt, user_prompt = self.build_prompt(state)
        last_error: Exception | None = None
        last_failure_type = "api_error"

        for attempt in range(self.max_retries + 1):
            try:
                return self._call_with_evidence(system_prompt, user_prompt, cache_bypass=use_bypass)
            except TeacherClientError as exc:
                if exc.failure_type == "config_error":
                    raise
                last_error, last_failure_type = exc, exc.failure_type
            except (httpx.TimeoutException, httpx.HTTPError, OSError) as exc:
                last_error = exc
                last_failure_type = "timeout" if isinstance(exc, httpx.TimeoutException) else "api_error"
            if attempt < self.max_retries:
                time.sleep(1.0 * (attempt + 1))

        raise TeacherClientError(
            last_failure_type, f"teacher API call failed after {self.max_retries} retries: {last_error}"
        )

    def _call_with_evidence(
        self, system_prompt: str, user_prompt: str, cache_bypass: bool = False
    ) -> dict:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if cache_bypass:
            # LiteLLM gateway cache control. Disables response caching for this
            # request only; never forwarded to the model as part of the prompt.
            payload["cache"] = {"no-cache": True}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        request_start = time.time()
        with httpx.Client(timeout=self.timeout, trust_env=False) as client:
            resp = client.post(url, json=payload, headers=headers)
            elapsed = time.time() - request_start
            resp.raise_for_status()
            data = resp.json()

        choices = data.get("choices") or []
        if not choices:
            raise TeacherClientError("api_error", "teacher returned no choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if content is None or not str(content).strip():
            # Reasoning models may exhaust the token budget before emitting the
            # final answer; treat as a retryable failure.
            raise TeacherClientError("empty_content", "teacher returned empty content")

        evidence = self._extract_evidence(resp, data, elapsed)
        return {"content": str(content), "evidence": evidence}

    @staticmethod
    def _extract_evidence(resp: httpx.Response, data: dict, elapsed: float) -> dict:
        """Collect non-secret request/response evidence for the cache diagnostic."""
        safe_headers = {
            "x-request-id",
            "x-litellm-call-id",
            "x-litellm-model-id",
            "x-litellm-response-duration-ms",
            "x-litellm-attempted-retries",
            "x-litellm-attempted-fallbacks",
            "x-upstream",
            "x-upstream-status",
            "x-cache",
            "x-cache-status",
            "x-cache-hit",
            "llm_provider-eo-cache-status",
            "llm_provider-x-ds-trace-id",
            "llm_provider-eo-log-uuid",
            "via",
            "server",
        }
        response_headers = {
            k: v for k, v in resp.headers.items() if k.lower() in safe_headers
        }
        choices = data.get("choices") or []
        return {
            "model": data.get("model") or "unknown",
            "completion_id": data.get("id"),
            "created": data.get("created"),
            "system_fingerprint": data.get("system_fingerprint"),
            "finish_reason": choices[0].get("finish_reason") if choices else None,
            "usage": data.get("usage"),
            "http_status": resp.status_code,
            "elapsed_seconds": round(elapsed, 4),
            "response_headers": response_headers,
        }


class MockTeacherClient:
    """Deterministic rule-based teacher for offline pipeline testing.

    NOT the research teacher. Rules are simple heuristics (heavy rain lowers
    bike preference, congestion lowers car preference, ...).
    """

    model = "mock_teacher"

    def respond(self, state: UniversalTravelerState) -> str:
        available = state.available_modes
        if not available:
            raise TeacherClientError("validation_error", "no available modes")

        # Start uniform over available modes.
        probs = {m: 1.0 for m in available}
        reasons: list[str] = []

        ctx = state.context
        if ctx.weather.intensity > 0.5:
            reasons.append("weather")
            for m in ("bike", "walk"):
                if m in probs:
                    probs[m] *= 0.3
        if ctx.road_congestion > 0.6:
            reasons.append("congestion")
            if "car" in probs:
                probs["car"] *= 0.5
        if ctx.transit_delay_min > 10 or ctx.transit_disruption:
            reasons.append("transit_disruption")
            if "pt" in probs:
                probs["pt"] *= 0.4
        if ctx.fare_multiplier > 1.3:
            reasons.append("fare")
            if "pt" in probs:
                probs["pt"] *= 0.7
        if ctx.road_disruption:
            reasons.append("road_disruption")
            if "car" in probs:
                probs["car"] *= 0.4

        # Normalize to sum exactly 1.0.
        total = sum(probs.values())
        if total <= 0:
            probs = {m: 1.0 / len(available) for m in available}
        else:
            probs = {m: v / total for m, v in probs.items()}

        selected = max(probs, key=probs.get)
        max_prob = probs[selected]
        confidence = round(0.5 + 0.5 * max_prob, 4)

        action = {
            "selected_mode": selected,
            "mode_probabilities": {m: round(p, 4) for m, p in probs.items()},
            "departure_time_shift_min": 0,
            "confidence": confidence,
            "reason_codes": reasons,
        }
        return json.dumps(action, ensure_ascii=False)
