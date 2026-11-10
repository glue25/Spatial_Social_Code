"""Only adapt the legacy request envelope to the provisioned native gateway.

No prompt suffix, JSON schema, MIME constraint, thinking override or output repair.
Archived legacy text extraction and 5-attempt transient/empty-text policy retained.
"""
import base64
import copy
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import requests

from egomas.src.jdcloud_gemini import _extract_text, GeminiServiceError


class Paused(RuntimeError):
    pass


class ModelIdentityError(RuntimeError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def native_body(parts, cap, model):
    legacy = {'model': model, 'contents': [{'role': 'user', 'parts': copy.deepcopy(parts)}],
              'generationConfig': {'temperature': 0.0, 'maxOutputTokens': cap}}
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

    def generate_parts(self, parts, max_output_tokens=4096):
        self.counter += 1
        folder = self.sample / f'{self.counter:02d}'
        legacy, body = native_body(parts, max_output_tokens, self.settings['model'])
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
                       prompt_or_generation_settings_changed=False)
        request_path = folder / 'request.json'
        if request_path.exists():
            assert json.loads(request_path.read_text()) == request, 'cached request mismatch'
        else:
            write(request_path, request)
        response_path = folder / 'response.json'
        if response_path.exists():
            raw = json.loads(response_path.read_text())
            assert raw.get('modelVersion') == self.settings['model']
            return _extract_text(raw)
        last = 'no response'
        for attempt in range(1, 6):
            if (self.out / 'OPERATOR_PAUSED.json').exists():
                raise Paused('operator pause')
            started = time.monotonic()
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
                    text = _extract_text(raw)
                    if not text:
                        raise GeminiServiceError('response contained no candidate text')
                    write(response_path, raw)
                    return text
                last = f'HTTP {response.status_code}'
                if response.status_code not in {408, 409, 429, 500, 502, 503, 504}:
                    break
            except (requests.RequestException, ValueError, GeminiServiceError) as exc:
                last = type(exc).__name__
                write(folder / f'exception_{attempt:02d}_{time.time_ns()}.json', {'error_type': last})
            if attempt < 5:
                time.sleep(min(30, 2 ** (attempt - 1)))
        raise GeminiServiceError('Legacy request policy exhausted: ' + last)
