"""Client for the LLM API used by the classification, with rate limiting and retries.

The provider and the model are chosen in the configuration cell of 02_Classification.ipynb; this module
only implements the HTTP calls. Each provider has its own _generate_<provider> method that returns the
same dict (text, model_version, usage, finish_reason, seconds); to use another API, add a method and
register it in PROVIDERS. Implemented: 'gemini' (Google AI Studio) and 'openai' (Chat Completions).

The API key is read from the environment variable LLM_API_KEY or from a .env file at the repository root
(never write it in notebooks or code).

Quotas: the client waits min_interval seconds between calls, follows the retry delay sent with a rate-limit
answer (HTTP 429), and raises QuotaExceeded when the daily quota is over, so the run can resume later.
"""
import json
import os
import re
import time

import requests



class QuotaExceeded(RuntimeError):
    """The daily request quota is over; resume the run on the next day."""


class LLMError(RuntimeError):
    pass


def load_api_key(env_file='.env'):
    key = os.environ.get('LLM_API_KEY')
    if not key and os.path.exists(env_file):
        for line in open(env_file, encoding='utf-8-sig'):
            m = re.match(r'\s*LLM_API_KEY\s*=\s*["\']?([^"\'\s]+)', line)
            if m:
                key = m.group(1)
    if not key:
        raise LLMError('Set LLM_API_KEY in the environment or in a .env file at the repository root.')
    return key


class LLMClient:
    def __init__(self, provider, model, api_key, temperature=0.0, max_output_tokens=32768,
                 response_schema=None, min_interval=7.0, timeout=300, max_retries=6, thinking_budget=None,
                 reasoning_effort=None):
        if provider not in PROVIDERS:
            raise LLMError(f'provider {provider!r} not implemented; available: {sorted(PROVIDERS)}')
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.response_schema = response_schema
        self.min_interval = min_interval
        self.timeout = timeout
        self.max_retries = max_retries
        self.thinking_budget = thinking_budget
        self.reasoning_effort = reasoning_effort
        self._last_call = 0.0
        self.schema_disabled = False

    def generate(self, system, user):
        """Send one request; returns dict(text, model_version, usage, finish_reason, seconds)."""
        return getattr(self, PROVIDERS[self.provider])(system, user)

    # ------------------------------------------------------------------ Gemini API (Google AI Studio)
    GEMINI_BASE = 'https://generativelanguage.googleapis.com/v1beta'

    def _gemini_payload(self, system, user):
        cfg = {'temperature': self.temperature, 'maxOutputTokens': self.max_output_tokens,
               'responseMimeType': 'application/json'}
        if self.response_schema and not self.schema_disabled:
            cfg['responseSchema'] = self.response_schema
        if self.thinking_budget is not None:
            cfg['thinkingConfig'] = {'thinkingBudget': self.thinking_budget}
        return {'systemInstruction': {'parts': [{'text': system}]},
                'contents': [{'role': 'user', 'parts': [{'text': user}]}],
                'generationConfig': cfg}

    def _wait(self):
        dt = time.time() - self._last_call
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        self._last_call = time.time()

    @staticmethod
    def _retry_delay(err):
        for d in err.get('details', []) or []:
            if str(d.get('@type', '')).endswith('RetryInfo'):
                m = re.match(r'([\d.]+)s', str(d.get('retryDelay', '')))
                if m:
                    return float(m.group(1))
        return None

    @staticmethod
    def _is_daily(err):
        text = json.dumps(err).lower()
        return 'perday' in text or 'per day' in text or 'requests_per_day' in text

    def _generate_gemini(self, system, user):
        url = f'{self.GEMINI_BASE}/models/{self.model}:generateContent'
        headers = {'Content-Type': 'application/json', 'X-goog-api-key': self.api_key}
        backoff = 15
        for attempt in range(self.max_retries + 1):
            self._wait()
            t0 = time.time()
            try:
                r = requests.post(url, headers=headers, json=self._gemini_payload(system, user), timeout=self.timeout)
            except requests.RequestException as e:          # network problem: wait and try again
                print(f'  network error ({e}); retrying in {backoff}s')
                time.sleep(backoff); backoff = min(backoff * 2, 300)
                continue
            if r.status_code == 200:
                data = r.json()
                cand = (data.get('candidates') or [{}])[0]
                parts = (cand.get('content') or {}).get('parts') or []
                text = ''.join(p.get('text', '') for p in parts if not p.get('thought'))
                return {'text': text, 'model_version': data.get('modelVersion', self.model),
                        'usage': data.get('usageMetadata', {}), 'finish_reason': cand.get('finishReason'),
                        'seconds': round(time.time() - t0, 1)}
            try:
                err = r.json().get('error', {})
            except ValueError:
                err = {'message': r.text[:500]}
            msg = err.get('message', '')
            if r.status_code == 429:
                if self._is_daily(err):
                    raise QuotaExceeded(msg)
                delay = self._retry_delay(err) or backoff
                print(f'  rate limit (429); waiting {delay:.0f}s')
                time.sleep(delay + 1); backoff = min(backoff * 2, 300)
                continue
            if r.status_code == 400 and self.response_schema and not self.schema_disabled and 'schema' in msg.lower():
                print(f'  the model rejected the response schema ({msg[:120]}); retrying without it')
                self.schema_disabled = True
                continue
            if r.status_code in (500, 502, 503, 504):
                print(f'  server error {r.status_code}; retrying in {backoff}s')
                time.sleep(backoff); backoff = min(backoff * 2, 300)
                continue
            raise LLMError(f'HTTP {r.status_code}: {msg}')
        raise LLMError('too many failed attempts')


    # ------------------------------------------------------------------ OpenAI API (Chat Completions)
    OPENAI_URL = 'https://api.openai.com/v1/chat/completions'

    @staticmethod
    def openai_schema(schema):
        """Convert prompt.RESPONSE_SCHEMA (Gemini subset, upper-case types) to a strict JSON schema:
        every property required, no extra properties, nullable fields accept null."""
        def conv(node):
            if not isinstance(node, dict):
                return node
            out = {}
            nullable = node.get('nullable', False)
            if 'type' in node:
                t = node['type'].lower()
                out['type'] = [t, 'null'] if nullable else t
            for k, v in node.items():
                if k in ('type', 'nullable', 'propertyOrdering', 'required'):
                    continue
                if k == 'properties':
                    out['properties'] = {name: conv(sub) for name, sub in v.items()}
                elif k == 'items':
                    out['items'] = conv(v)
                elif k == 'enum':
                    out['enum'] = list(v) + ([None] if nullable else [])
                else:
                    out[k] = v
            if 'properties' in out:
                out['additionalProperties'] = False
                out['required'] = list(out['properties'].keys())
            return out
        return conv(schema)

    @staticmethod
    def _retry_after(headers, default):
        try:
            return float(headers.get('retry-after', default))
        except (TypeError, ValueError):
            return default

    def _generate_openai(self, system, user):
        headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}
        payload = {'model': self.model,
                   'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
                   'max_completion_tokens': self.max_output_tokens}
        if self.temperature is not None:          # reasoning models accept only the default temperature
            payload['temperature'] = self.temperature
        if self.reasoning_effort:
            payload['reasoning_effort'] = self.reasoning_effort
        if self.response_schema:
            payload['response_format'] = {'type': 'json_schema', 'json_schema': {
                'name': 'classification', 'strict': True, 'schema': self.openai_schema(self.response_schema)}}
        backoff = 5
        for attempt in range(self.max_retries + 1):
            self._wait()
            t0 = time.time()
            try:
                r = requests.post(self.OPENAI_URL, headers=headers, json=payload, timeout=self.timeout)
            except requests.RequestException as e:
                print(f'  network error ({e}); retrying in {backoff}s')
                time.sleep(backoff); backoff = min(backoff * 2, 300)
                continue
            if r.status_code == 200:
                data = r.json()
                choice = (data.get('choices') or [{}])[0]
                message = choice.get('message') or {}
                usage = data.get('usage') or {}
                details = usage.get('completion_tokens_details') or {}
                return {'text': message.get('content') or '', 'model_version': data.get('model', self.model),
                        'usage': {'totalTokenCount': usage.get('total_tokens', 0),
                                  'promptTokenCount': usage.get('prompt_tokens', 0),
                                  'candidatesTokenCount': usage.get('completion_tokens', 0),
                                  'thoughtsTokenCount': details.get('reasoning_tokens', 0)},
                        'finish_reason': choice.get('finish_reason'),
                        'seconds': round(time.time() - t0, 1)}
            try:
                err = r.json().get('error', {})
            except ValueError:
                err = {'message': r.text[:500]}
            msg = err.get('message', '')
            if r.status_code == 429:
                if err.get('code') == 'insufficient_quota':
                    raise LLMError(f'no credit left on the OpenAI account: {msg}')
                delay = self._retry_after(r.headers, backoff)
                print(f'  rate limit (429); waiting {delay:.0f}s')
                time.sleep(delay + 1); backoff = min(backoff * 2, 300)
                continue
            if r.status_code in (500, 502, 503, 504):
                print(f'  server error {r.status_code}; retrying in {backoff}s')
                time.sleep(backoff); backoff = min(backoff * 2, 300)
                continue
            raise LLMError(f'HTTP {r.status_code}: {msg}')
        raise LLMError('too many failed attempts')


PROVIDERS = {'gemini': '_generate_gemini', 'openai': '_generate_openai'}
