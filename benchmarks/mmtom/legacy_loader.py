"""Load archived code unchanged; disable the unused local Qwen backend."""
from pathlib import Path
import sys
import types

HOME = Path(__file__).resolve().parent
SOURCE = HOME / 'historical_source'


def load():
    sys.path[:0] = [str(SOURCE / 'MA-EgoQA'), str(SOURCE / 'spatial_social_code'),
                   str(SOURCE / 'AutoToM/model')]
    # evaluate_spatial_social imports a local GPU backend at module scope, even
    # when the API runner uses only its prompt/normalizer. Never substitute a
    # local model: any accidental use of this backend is a hard error.
    backend = types.ModuleType('local_backend')
    def forbidden(*args, **kwargs):
        raise RuntimeError('Local Qwen backend is forbidden in the model-only API rerun')
    for name in ('choice_probabilities', 'generate', 'set_image_context'):
        setattr(backend, name, forbidden)
    sys.modules['local_backend'] = backend

