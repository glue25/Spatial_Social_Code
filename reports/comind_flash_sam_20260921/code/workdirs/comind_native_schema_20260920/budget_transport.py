"""CoMind native JSON schema envelope; same capped, persistent transport policy.
Separate copy: never edit the source frozen by the active MA workers.
"""
import base64
import copy
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import requests

from egomas.src.jdcloud_gemini import _extract_text, GeminiServiceError


class OutputTruncated(RuntimeError):
    pass


class Paused(RuntimeError):
    pass


class ModelIdentityError(RuntimeError):
    pass


class PermanentServiceError(RuntimeError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def native_body(parts, cap, model, response_schema=None):
    legacy = {'model': model, 'contents': [{'role': 'user', 'parts': copy.deepcopy(parts)}],
              'generationConfig': {'temperature': 0.0, 'maxOutputTokens': cap}}
    if response_schema is not None:
        legacy['generationConfig'].update(responseMimeType='application/json',
                                         responseJsonSchema=copy.deepcopy(response_schema))
    native = copy.deepcopy(legacy)
    del native['model']  # model is selected in the native URL
    for part in native['contents'][0]['parts']:
        if 'inline_data' in part:
            part['inlineData'] = part.pop('inline_data')
    return legacy, native


class Client:
    def __init__(self, out, config, gate):
        self.out, self.settings, self.gate = out, config, gate
        self.key = Path(config['key_file']).read_text().strip()
        assert self.key
        self.config = SimpleNamespace(model=config['model'])
        self.counter = 0
        self.sample = None

    def begin(self, index):
        self.counter = 0
        self.sample = self.out / 'calls' / f'{index:04d}'

    def generate(self, prompt, max_output_tokens=4096):
        return self.generate_parts([{'text': prompt}], max_output_tokens)

    def generate_parts(self, parts, max_output_tokens=4096, response_schema=None):
        legacy_requested_cap = max_output_tokens
        max_output_tokens = self.settings["uniform_output_cap"]
        self.counter += 1
        folder = self.sample / f'{self.counter:02d}'
        legacy, body = native_body(parts, max_output_tokens, self.settings['model'], response_schema)
        safe_parts = []
        for part in parts:
            if 'inline_data' in part:
                media = part['inline_data']; data = base64.b64decode(media['data'])
                safe_parts.append({'mimeType': media['mimeType'], 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
            else:
                safe_parts.append(part)
        request = dict(parts=safe_parts, generationConfig=legacy['generationConfig'],
                       model=self.settings['model'], signature=digest(legacy),
                       envelope_changes=['model moved to URL', 'inline_data renamed inlineData'],
                       prompt_or_generation_settings_changed=True, prompts_changed=False,
                       schema_transport='native_responseJsonSchema' if response_schema is not None else None,
                       legacy_requested_cap=legacy_requested_cap, budget_adjustment="uniform cap for Direct and SSC")
        request_path = folder / 'request.json'
        if request_path.exists():
            assert json.loads(request_path.read_text()) == request, 'cached request mismatch'
        else:
            write(request_path, request)
        response_path = folder / 'response.json'
        if (folder / 'exhausted.json').exists():
            raise GeminiServiceError('Preserved exhausted transport request; no outcome-based resampling')
        if (folder / 'incomplete.json').exists():
            raise OutputTruncated('Preserved non-STOP response; no resampling')
        if response_path.exists():
            raw = json.loads(response_path.read_text())
            assert raw.get('modelVersion') == self.settings['model']
            if any(c.get('finishReason') != 'STOP' for c in raw.get('candidates', [])):
                raise OutputTruncated('Cached response is incomplete')
            return _extract_text(raw)
        last = 'no response'
        max_attempts = self.settings.get("transport_max_attempts", 10)
        used = [int(m.group(1)) for p in folder.glob('*.json')
                if (m := re.match(r'(?:attempt|exception)_(\d+)_', p.name))]
        for attempt in range(max(used, default=0) + 1, max_attempts + 1):
            if (self.out / 'OPERATOR_PAUSED.json').exists():
                raise Paused('operator pause')
            started = time.monotonic()
            write(self.out / "transport_status.json", {"utc": datetime.now(timezone.utc).isoformat(), "state": "request_pending", "sample": self.sample.name, "call": self.counter, "attempt": attempt, "max_attempts": max_attempts})
            try:
                with self.gate(self.settings['shared_gate'], 15):
                    if (self.out / 'OPERATOR_PAUSED.json').exists():
                        raise Paused('operator pause')
                    response = requests.post(self.settings['endpoint'], json=body,
                        headers={'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'},
                        timeout=600, allow_redirects=False)
                try:
                    raw = response.json()
                except ValueError:
                    raw = {'non_json_response': response.text[:1000]}
                raw = json.loads(json.dumps(raw).replace(self.key, '[REDACTED]'))
                write(folder / f'attempt_{attempt:02d}_{time.time_ns()}.json', dict(http_status=response.status_code,
                    elapsed_seconds=time.monotonic()-started, response=raw))
                if response.status_code == 200:
                    if not isinstance(raw, dict) or raw.get('modelVersion') != self.settings['model']:
                        raise ModelIdentityError('Unexpected provider-reported model identity; no fallback')
                    if any(c.get('finishReason') != 'STOP' for c in raw.get('candidates', [])):
                        write(folder / 'incomplete.json', {'reason': [c.get('finishReason') for c in raw.get('candidates', [])], 'cap': max_output_tokens})
                        raise OutputTruncated('Incomplete output; no partial-code execution or cap escalation')
                    text = _extract_text(raw)
                    if not text:
                        raise GeminiServiceError('response contained no candidate text')
                    write(response_path, raw)
                    return text
                last = f'HTTP {response.status_code}'
                if response.status_code not in {408, 409, 429, 500, 502, 503, 504}:
                    raise PermanentServiceError('Nonretryable service HTTP ' + str(response.status_code))
            except (requests.RequestException, ValueError, GeminiServiceError) as exc:
                last = type(exc).__name__
                chain = []
                current = exc
                seen = set()
                while current is not None and id(current) not in seen:
                    seen.add(id(current))
                    chain.append({'error_type': type(current).__name__, 'message': str(current).replace(self.key, '[REDACTED]')})
                    current = current.__cause__ or current.__context__
                write(folder / f'exception_{attempt:02d}_{time.time_ns()}.json', {'utc': datetime.now(timezone.utc).isoformat(), 'error_type': last, 'chain': chain, 'elapsed_seconds': time.monotonic()-started})
            if attempt < max_attempts:
                delay = min(120, 5 * 2 ** (attempt - 1))
                write(self.out / "transport_status.json", {"utc": datetime.now(timezone.utc).isoformat(), "state": "retry_wait", "sample": self.sample.name, "call": self.counter, "attempt": attempt, "max_attempts": max_attempts, "delay_seconds": delay, "last_error": last})
                time.sleep(delay)
        write(folder / 'exhausted.json', {'max_attempts': max_attempts, 'last_error': last})
        raise GeminiServiceError(f'Bounded transport policy exhausted after {max_attempts} attempts: ' + last)
