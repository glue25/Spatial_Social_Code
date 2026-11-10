"""Transport-only guard used by the isolated Flash Habitat source snapshot."""
import json
import os
from pathlib import Path

MODEL = 'gemini-3-flash-preview'


def request_options(options):
    out = Path(os.environ['QDD_HABITAT_OUTPUT'])
    if (out / 'OPERATOR_PAUSED.json').exists():
        raise RuntimeError('QDD Habitat operator pause')
    if options.get('model') != MODEL:
        raise RuntimeError('QDD Habitat disallows secondary model calls')
    return {**options, 'reasoning_effort': 'low', 'max_tokens': 8192,
            'timeout': 180}


def validate_and_log(response):
    out = Path(os.environ['QDD_HABITAT_OUTPUT'])
    raw = response.model_dump()
    key = os.environ.get('GEMINI_API_KEY', '')
    serialized = json.dumps(raw, ensure_ascii=False)
    if key: serialized = serialized.replace(key, '[REDACTED]')
    with (out / 'api_responses.jsonl').open('a') as f:
        f.write(serialized + '\n')
    if response.model != MODEL:
        raise RuntimeError('QDD Habitat returned wrong model identity')
    if len(response.choices) != 1 or response.choices[0].finish_reason != 'stop':
        raise RuntimeError('QDD Habitat response incomplete')
    if not response.choices[0].message.content:
        raise RuntimeError('QDD Habitat returned no planner text')
