"""Checks for the LLM client: strict JSON schema for OpenAI and the OpenAI request/response mapping
(no network: requests.post is replaced). Run from the repository root: python tests/test_llm_client.py"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import requests  # noqa: E402
from mdm import llm_client, prompt as cp  # noqa: E402


def _walk(node, path='root'):
    """Yield (path, node) for every object schema, to check the rules of strict mode."""
    if isinstance(node, dict):
        if 'properties' in node:
            yield path, node
            for k, v in node['properties'].items():
                yield from _walk(v, f'{path}.{k}')
        if 'items' in node:
            yield from _walk(node['items'], f'{path}[]')


def test_openai_schema_is_strict():
    s = llm_client.LLMClient.openai_schema(cp.RESPONSE_SCHEMA)
    objects = list(_walk(s))
    assert len(objects) == 2                       # top level and one result item
    for path, node in objects:
        assert node['additionalProperties'] is False, path
        assert set(node['required']) == set(node['properties']), path
    item = s['properties']['results']['items']
    assert item['properties']['marketing_mix']['type'] == ['string', 'null']
    assert None in item['properties']['marketing_mix']['enum']
    assert item['properties']['ai_techniques']['type'] == 'array'
    assert 'propertyOrdering' not in json.dumps(s) and 'nullable' not in json.dumps(s)


def test_openai_request_and_response(monkeypatch=None):
    sent = {}

    class Resp:
        status_code = 200
        headers = {}
        text = ''

        def json(self):
            return {'model': 'gpt-5-nano-2025-08-07',
                    'choices': [{'finish_reason': 'stop', 'message': {'content': '{"results": []}'}}],
                    'usage': {'prompt_tokens': 100, 'completion_tokens': 50, 'total_tokens': 150,
                              'completion_tokens_details': {'reasoning_tokens': 30}}}

    def fake_post(url, headers=None, json=None, timeout=None):
        sent.update(url=url, headers=headers, body=json)
        return Resp()

    real = requests.post
    requests.post = fake_post
    try:
        c = llm_client.LLMClient('openai', 'gpt-5-nano', 'test-key', temperature=None, min_interval=0,
                                 response_schema=cp.RESPONSE_SCHEMA, reasoning_effort='low')
        out = c.generate('SYSTEM', 'USER')
    finally:
        requests.post = real
    body = sent['body']
    assert sent['url'] == 'https://api.openai.com/v1/chat/completions'
    assert sent['headers']['Authorization'] == 'Bearer test-key'
    assert body['model'] == 'gpt-5-nano' and 'temperature' not in body
    assert body['reasoning_effort'] == 'low'
    assert body['response_format']['json_schema']['strict'] is True
    assert body['messages'] == [{'role': 'system', 'content': 'SYSTEM'}, {'role': 'user', 'content': 'USER'}]
    assert out['text'] == '{"results": []}' and out['model_version'] == 'gpt-5-nano-2025-08-07'
    assert out['usage']['totalTokenCount'] == 150 and out['usage']['thoughtsTokenCount'] == 30


def test_unknown_provider():
    try:
        llm_client.LLMClient('nope', 'm', 'k')
    except llm_client.LLMError:
        return
    raise AssertionError('unknown provider accepted')


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
