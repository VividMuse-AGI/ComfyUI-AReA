"""Build a local review archive. No git, GitHub API or server calls."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from verify_release import ROOT, files, sha, verify

def build(output_dir):
    verify()
    version = (ROOT / 'VERSION.txt').read_text().strip()
    manifest = {
        'version': version,
        'suggested_tag': f'v{version}',
        'release_kind': 'experimental prerelease',
        'status': 'experimental prerelease',
        'scope': 'Documentation, portable annotated workflows and existing test evidence; inference code unchanged.',
        'gpu_generation_during_preparation': False,
        'remote_ci_run_claimed': False,
        'upstream_area_commit': '9e38ec2d8457f2aa7800766e8a22d4960f160c69',
        'comfy_reference_commit': 'ba994879b88d0ebd9950b1e5c1e923d928f216ba',
        'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in files() if p.name != 'MANIFEST.json'},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f'ComfyUI-AReA-v{version}.zip'
    if archive.exists() or archive.with_suffix('.zip.sha256').exists():
        raise FileExistsError('Review archive already exists; choose a new output directory.')
    (ROOT / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    verify(check_manifest=True)
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for rel in sorted([*manifest['files'], 'MANIFEST.json']):
            info = zipfile.ZipInfo('ComfyUI-AReA/' + rel, date_time=(2026, 10, 15, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, (ROOT / rel).read_bytes())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('ZIP CRC verification failed')
        if len(z.namelist()) != len(manifest['files']) + 1:
            raise ValueError('ZIP file count mismatch')
        for rel, expected in manifest['files'].items():
            if hashlib.sha256(z.read('ComfyUI-AReA/' + rel)).hexdigest() != expected:
                raise ValueError(f'ZIP hash mismatch: {rel}')
    checksum = sha(archive)
    archive.with_suffix('.zip.sha256').write_text(f'{checksum}  {archive.name}\n', encoding='utf-8')
    print(json.dumps({'archive': archive.name, 'bytes': archive.stat().st_size,
                      'files': len(manifest['files']) + 1, 'sha256': checksum,
                      'crc_and_embedded_hashes': 'passed', 'published': False}, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'dist')
    build(parser.parse_args().output_dir.resolve())
