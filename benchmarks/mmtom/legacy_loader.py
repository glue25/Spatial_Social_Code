"""Load historical prompts with shared helpers and the explicit legacy runtime."""
from pathlib import Path
import sys
import types

HOME = Path(__file__).resolve().parent
ROOT = HOME.parents[1]


def load():
    sys.path[:0] = [str(HOME / 'legacy'), str(ROOT),
                   str(ROOT / 'benchmarks/ma_egoqa'),
                   str(ROOT / 'spatial_social_code'),
                   str(ROOT / 'third_party/autotom/model')]
    backend = types.ModuleType('local_backend')
    def forbidden(*args, **kwargs):
        raise RuntimeError('Local Qwen backend is forbidden in the model-only API rerun')
    for name in ('choice_probabilities', 'generate', 'set_image_context'):
        setattr(backend, name, forbidden)
    sys.modules['local_backend'] = backend
