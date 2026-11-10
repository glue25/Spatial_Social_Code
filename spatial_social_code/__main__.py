"""Repository commands; run `python -m spatial_social_code --help` from the checkout."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / '.github/release-manifest.json'
ENTRYPOINTS = {
    'mmtom': 'benchmarks/mmtom/matched_runner.py',
    'ma-egoqa': 'benchmarks/ma_egoqa/egomas/src/run_video_provenance_v3_jdcloud.py',
    'comind': 'benchmarks/comind/pipelines/compiler/run.py',
    'habitat': 'benchmarks/habitat/run.py',
    'information-history': '-m benchmarks.information_history.run',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    return dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=os.pathsep.join(
        str(ROOT / p) for p in ('.', 'benchmarks/ma_egoqa', 'spatial_social_code')))


def release_files():
    result = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=ROOT)
    return sorted({s for s in result.decode().split('\0')
                   if s and s != MANIFEST.relative_to(ROOT).as_posix() and (ROOT / s).is_file()})


def manifest():
    metadata = json.loads(MANIFEST.read_text())
    metadata['files'] = {s: sha(ROOT / s) for s in release_files()}
    MANIFEST.write_text(json.dumps(metadata, indent=2) + '\n')
    print(f"Recorded {len(metadata['files'])} source hashes")


def check():
    metadata = json.loads(MANIFEST.read_text())
    for name, digest in metadata['files'].items():
        path = ROOT / name
        if sha(path) != digest:
            raise ValueError(f'Modified release file: {name}')
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=name)
    print(f"Verified {len(metadata['files'])} release files and Python syntax")


def demo():
    from spatial_social_code.runtime import execute_spatial, execute_social
    spatial = execute_spatial('S.fact(0, "inside", "cup", "cabinet", source="example", evidence_id="S1")')
    social = execute_social('M.belief(0, "Alice", "The cup is inside the cabinet.", 1.0, "Hand-written premise", evidence_ids="S1")',
                            valid_evidence_ids={'S1'}, require_evidence=True)
    print(json.dumps({'spatial': [json.loads(s) for s in spatial.splitlines()],
                      'social': [json.loads(s) for s in social.splitlines()]}, indent=2))


def test():
    suites = [(ROOT, ['tests', 'benchmarks/ma_egoqa/egomas/src']),
              (ROOT / 'benchmarks/comind/pipelines/sam', ['test_sidecar.py'])]
    for cwd, targets in suites:
        subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *targets],
                       cwd=cwd, env=environment(), check=True)

    subprocess.run([sys.executable, 'test_native_schema.py'],
                   cwd=ROOT / 'benchmarks/comind/pipelines/compiler',
                   env=environment(), check=True)

def verify_habitat(destination, config):
    for name, digest in config['expected_files'].items():
        if not (destination / name).is_file() or sha(destination / name) != digest:
            raise ValueError(f'EnactToM file differs from submission: {name}')


def setup_habitat(archive=None, destination=None):
    """Reconstruct the submitted dependency without vendoring 218 source files."""
    config = json.loads(MANIFEST.read_text())['enacttom']
    destination = Path(destination or ROOT / 'external/EnactTom').resolve()
    if destination.exists():
        verify_habitat(destination, config)
        print(f'Existing EnactToM source verified: {destination}')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.ssc-enacttom-', dir=destination.parent) as temp:
        temp = Path(temp)
        archive = Path(archive).resolve() if archive else temp / 'upstream.tar.gz'
        if not archive.exists():
            if archive != temp / 'upstream.tar.gz':
                raise FileNotFoundError(archive)
            with urllib.request.urlopen(config['url'], timeout=120) as response:
                archive.write_bytes(response.read())
        if sha(archive) != config['archive_sha256']:
            raise ValueError('Upstream archive checksum mismatch')
        source = temp / 'source'
        source.mkdir()
        with tarfile.open(archive) as bundle:
            for member in bundle.getmembers():
                parts = Path(member.name).parts
                if not parts or Path(member.name).is_absolute() or '..' in parts:
                    raise ValueError('Unsafe archive path')
                if len(parts) == 1:
                    continue
                target = source.joinpath(*parts[1:])
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(bundle.extractfile(member).read())
                    target.chmod(0o755 if member.mode & 0o111 else 0o644)
                elif member.issym():
                    link = Path(member.linkname)
                    if link.is_absolute() or not (target.parent / link).resolve().is_relative_to(source.resolve()):
                        raise ValueError('Unsafe archive link')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.symlink_to(link)
                else:
                    raise ValueError(f'Unsupported archive member: {member.name}')
        subprocess.run(['patch', '-p1', '--batch', '--forward', '--fuzz=0',
                        '-i', str(ROOT / config['patch'])], cwd=source, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        verify_habitat(source, config)
        source.rename(destination)
    print(f'Reconstructed and verified {len(config["expected_files"])} submitted files: {destination}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('check', 'test', 'demo', 'manifest'):
        commands.add_parser(name)
    setup = commands.add_parser('setup-habitat', help='Fetch pinned upstream source and apply SSC changes')
    setup.add_argument('--archive', type=Path, help='Use an already downloaded upstream tar.gz')
    setup.add_argument('--destination', type=Path, help='Default: external/EnactTom')
    run = commands.add_parser('run', help='Forward arguments to a benchmark entry point')
    run.add_argument('benchmark', choices=ENTRYPOINTS)
    run.add_argument('args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == 'setup-habitat':
        setup_habitat(args.archive, args.destination)
    elif args.command == 'run':
        entry = ENTRYPOINTS[args.benchmark]
        tail = args.args[1:] if args.args[:1] == ['--'] else args.args
        argv = entry.split() if entry.startswith('-m ') else [str(ROOT / entry)]
        subprocess.run([sys.executable, *argv, *tail], cwd=ROOT, env=environment(), check=True)
    else:
        globals()[args.command]()


if __name__ == '__main__':
    main()
