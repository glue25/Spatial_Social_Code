from types import SimpleNamespace
import pytest
from benchmarks.habitat import guard


def test_only_flash_and_shared_budget(monkeypatch,tmp_path):
    monkeypatch.setenv('QDD_HABITAT_OUTPUT',str(tmp_path))
    opts=guard.request_options({'model':guard.MODEL,'messages':[],'max_tokens':256})
    assert opts['max_tokens']==8192 and opts['reasoning_effort']=='low' and opts['timeout']==180
    with pytest.raises(RuntimeError):guard.request_options({'model':'gpt-5.2'})
    (tmp_path/'OPERATOR_PAUSED.json').write_text('{}')
    with pytest.raises(RuntimeError):guard.request_options({'model':guard.MODEL})


def test_response_identity_and_secret_redaction(monkeypatch,tmp_path):
    monkeypatch.setenv('QDD_HABITAT_OUTPUT',str(tmp_path));monkeypatch.setenv('GEMINI_API_KEY','unit-secret')
    r=SimpleNamespace(model=guard.MODEL,choices=[SimpleNamespace(finish_reason='stop',
        message=SimpleNamespace(content='Wait[]'))],model_dump=lambda:{'model':guard.MODEL,'extra':'unit-secret'})
    guard.validate_and_log(r)
    assert 'unit-secret' not in (tmp_path/'api_responses.jsonl').read_text()
    r.model='gemini-3.1-pro-preview'
    with pytest.raises(RuntimeError):guard.validate_and_log(r)


def test_incomplete_response_rejected(monkeypatch,tmp_path):
    monkeypatch.setenv('QDD_HABITAT_OUTPUT',str(tmp_path))
    r=SimpleNamespace(model=guard.MODEL,choices=[SimpleNamespace(finish_reason='length')],model_dump=lambda:{})
    with pytest.raises(RuntimeError):guard.validate_and_log(r)
