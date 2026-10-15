"""Read-only static verification. No torch, network, server or model execution."""
import argparse
import ast
import copy
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {'.git', '__pycache__', '.pytest_cache', '.venv', '.comfy-reference', 'outputs', 'dist'}
WIDGETS = {
    1: ['unet_name', 'weight_dtype'],
    2: ['max_shift', 'base_shift', 'width', 'height'],
    3: ['enabled', 'phase1_steps', 'pooling', 'halo', 'bridge_ratio', 'seed', 'window_batch_size'],
    4: ['clip_name1', 'clip_name2', 'type', 'device'], 5: ['text'], 6: ['guidance'], 7: ['text'],
    8: ['width', 'height', 'batch_size'],
    9: ['seed', None, 'steps', 'cfg', 'sampler_name', 'scheduler', 'denoise'],
    10: ['vae_name'], 11: ['tile_size', 'overlap', 'temporal_size', 'temporal_overlap'],
    12: ['filename_prefix'],
}

def files(root=ROOT):
    return sorted(p for p in root.rglob('*') if p.is_file()
                  and not any(part in IGNORED for part in p.relative_to(root).parts)
                  and not p.name.endswith(('.pyc', '.log')))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def require(condition, message):
    if not condition:
        raise ValueError(message)

def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from strings(key)
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)

def verify_workflow(path):
    graph = read(path)
    api = read(path.with_name(path.stem + '_API.json'))
    nodes = {int(n['id']): n for n in graph['nodes']}
    require(len(nodes) == len(graph['nodes']) == 18, f'{path.name}: unique 18 nodes')
    require(set(nodes) == set(range(1, 19)), f'{path.name}: node IDs')
    require(set(api) == {str(i) for i in range(1, 13)}, f'{path.name}: 12 executable nodes')
    require(all(n.get('title') is None for n in nodes.values()), f'{path.name}: default node titles')
    require(all('widgets_values_named' not in n for n in nodes.values()), f'{path.name}: no stale named-widget extension data')
    require(all(n.get('mode', 0) == 0 for n in nodes.values()), f'{path.name}: nodes enabled, no bypass')
    require(len(graph['groups']) == 5, f'{path.name}: five annotation groups')
    require(all(nodes[i]['type'] == 'Note' for i in range(13, 19)), f'{path.name}: six Notes')
    for nid in range(13, 19):
        require(not nodes[nid].get('inputs') and not nodes[nid].get('outputs'), f'{path.name}: disconnected Notes')
        require(bool(nodes[nid]['widgets_values'][0].strip()), f'{path.name}: nonempty Notes')
    link_ids = set()
    linked_inputs = set()
    linked_outputs = {}
    for lid, src, slot, dst, index, typ in graph['links']:
        require(lid not in link_ids, f'{path.name}: unique link ID')
        link_ids.add(lid)
        require(src in WIDGETS and dst in WIDGETS, f'{path.name}: executable link endpoints')
        output, inp = nodes[src]['outputs'][slot], nodes[dst]['inputs'][index]
        require(output['type'] == inp['type'] == typ, f'{path.name}: link type')
        require(lid in output['links'] and inp['link'] == lid, f'{path.name}: bidirectional link')
        require(api[str(dst)]['inputs'][inp['name']] == [str(src), slot], f'{path.name}: API link')
        require((dst, index) not in linked_inputs, f'{path.name}: single input link')
        linked_inputs.add((dst, index))
        linked_outputs.setdefault((src, slot), set()).add(lid)
    for nid in WIDGETS:
        for index, inp in enumerate(nodes[nid].get('inputs', [])):
            if inp.get('link') is not None:
                require((nid, index) in linked_inputs, f'{path.name}: no orphan input link')
        for slot, output in enumerate(nodes[nid].get('outputs', [])):
            require(set(output.get('links') or []) == linked_outputs.get((nid, slot), set()), f'{path.name}: no orphan output link')
    for nid, keys in WIDGETS.items():
        require(nodes[nid]['type'] == api[str(nid)]['class_type'], f'{path.name}: class {nid}')
        require(len(nodes[nid]['widgets_values']) == len(keys), f'{path.name}: widget count {nid}')
        for i, key in enumerate(keys):
            if key is not None:
                require(nodes[nid]['widgets_values'][i] == api[str(nid)]['inputs'][key], f'{path.name}: widget {nid}.{key}')
    width = 2048 if '_2K_' in path.name else 4096
    enabled = not path.stem.endswith('_OFF')
    require(api['3']['inputs']['enabled'] is enabled, f'{path.name}: ON/OFF')
    require(nodes[9]['widgets_values'][1] == 'fixed', f'{path.name}: fixed seed control')
    for nid in ('2', '8'):
        require(api[nid]['inputs']['width'] == api[nid]['inputs']['height'] == width, f'{path.name}: paired dimensions')
    require(api['2']['inputs']['max_shift'] == api['2']['inputs']['base_shift'] == 1.15, f'{path.name}: shifts')
    require(api['9']['inputs']['seed'] == api['3']['inputs']['seed'] == 0, f'{path.name}: aligned seeds')
    require(api['9']['inputs']['steps'] == 28 and api['9']['inputs']['cfg'] == 1, f'{path.name}: sampling settings')
    require(api['9']['inputs']['sampler_name'] == 'euler' and api['9']['inputs']['scheduler'] == 'simple', f'{path.name}: sampler')
    require(api['9']['inputs']['denoise'] == 1 and api['6']['inputs']['guidance'] == 4.5, f'{path.name}: guidance')
    require(api['8']['inputs']['batch_size'] == 1, f'{path.name}: batch')
    require(api['3']['inputs']['phase1_steps'] == 0 and api['3']['inputs']['window_batch_size'] == 1, f'{path.name}: phase/windows')
    for nid, key in (('1', 'unet_name'), ('4', 'clip_name1'), ('4', 'clip_name2'), ('10', 'vae_name')):
        name = api[nid]['inputs'][key]
        require('/' not in name and '\\' not in name, f'{path.name}: portable model basename')
    require(set(graph['extra']) == {'ds'}, f'{path.name}: no extension/local model metadata')
    return api

def verify(root=ROOT, check_manifest=False):
    root = Path(root)
    paths = files(root)
    checked = []
    for path in paths:
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8'), filename=path.relative_to(root).as_posix())
        if path.suffix.lower() in {'.py', '.md', '.json', '.txt', '.yml', '.toml', '.csv'} or path.name in {'NOTICE', 'LICENSE', '.gitignore'}:
            raw = path.read_text(encoding='utf-8')
            values = list(strings(json.loads(raw))) if path.suffix == '.json' else [raw]
            for value in values:
                require(not re.search(r'(?<![A-Za-z0-9])[A-Za-z]:[\\/]', value), f'Absolute drive path in {path.relative_to(root)}')
                require(not re.search(r'/(?:Users|home)/[^/\s]+/', value), f'User home path in {path.relative_to(root)}')
                require(not re.search(r'https?://(?:127\.0\.0\.1|localhost):\d+', value), f'Local server endpoint in {path.relative_to(root)}')
                require('\ufffd' not in value, f'Unicode replacement character in {path.relative_to(root)}')
    checked.append('Python syntax, JSON parsing, portable paths and Unicode')
    license_record = read(root / 'validation/license_audit.json')
    license_bytes = (root / 'LICENSE').read_bytes()
    license_blob = hashlib.sha1(b'blob ' + str(len(license_bytes)).encode() + b'\0' + license_bytes).hexdigest()
    require(license_blob == license_record['upstream_license_git_blob_sha1'] == '261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64', 'LICENSE matches pinned upstream blob')
    require(sha(root / 'LICENSE') == license_record['license_sha256'], 'LICENSE recorded SHA256')
    notice = (root / 'NOTICE').read_text(encoding='utf-8')
    for name in license_record['upstream_authors']:
        require(name in notice, f'Upstream author in NOTICE: {name}')
    for rel in license_record['modified_upstream_files']:
        require(rel in notice, f'Modified upstream path in NOTICE: {rel}')
        require((root / rel).read_text(encoding='utf-8').startswith('# Modified for ComfyUI-AReA:'), f'Modification notice: {rel}')
    for rel in ('area_core/backend.py', 'area_core/__init__.py'):
        require(rel in notice, f'Added source attribution: {rel}')
    checked.append('Pinned upstream license, author attribution and modified-file notices')
    for path in root.rglob('*.md'):
        if path not in paths:
            continue
        for link in re.findall(r'!?\[[^\]\n]*\]\(([^)\s]+)\)', path.read_text(encoding='utf-8')):
            if re.match(r'[A-Za-z][A-Za-z0-9+.-]*:', link) or link.startswith('#'):
                continue
            target = (path.parent / unquote(link.split('#', 1)[0])).resolve()
            require(target.is_relative_to(root.resolve()) and target.exists(), f'Broken/internal link: {path.name} -> {link}')
    checked.append('Local Markdown links resolve within the repository')
    baseline = read(root / 'validation/inference_baseline.json')
    for rel, expected in baseline['files'].items():
        require(sha(root / rel) == expected, f'Inference/test baseline changed: {rel}')
    checked.append('Production and existing test Python hashes match tested baseline')
    for record in read(root / 'validation/historical_sources.json')['historical_files']:
        require(sha(root / record['public_file']) == record['public_sha256'], f'Historical record hash: {record["public_file"]}')
    checked.append('Normalized historical evidence hashes match provenance index')
    pairs = sorted(p for p in (root / 'workflows').rglob('*.json') if not p.stem.endswith('_API'))
    require(len(pairs) == 6, 'Six annotated UI/API pairs')
    apis = {p.relative_to(root / 'workflows').as_posix(): verify_workflow(p) for p in pairs}
    for name in ('01_Krea_FP8_2K_AReA_ON.json', '02_Krea_FP8_4K_AReA_ON.json', '03_Krea_FP8_4K_AReA_OFF.json'):
        require(apis[name] == apis['zh-CN/' + name], f'English/Chinese execution equivalence: {name}')
    on = copy.deepcopy(apis['02_Krea_FP8_4K_AReA_ON.json'])
    off = copy.deepcopy(apis['03_Krea_FP8_4K_AReA_OFF.json'])
    on['3']['inputs']['enabled'] = off['3']['inputs']['enabled'] = False
    on['12']['inputs']['filename_prefix'] = off['12']['inputs']['filename_prefix'] = ''
    require(on == off, '4K ON/OFF only differ in enabled and output prefix')
    checked.append('Six UI/API graphs, default titles, Notes/groups, links, widgets and bilingual/ON-OFF parity')
    require(not any('widgets_values_named' in n for p in pairs for n in read(p)['nodes']), 'No redundant named-widget data in public graphs')
    evidence = read(root / 'validation/4k28_repeat.json')
    runs = evidence['completed_runs']
    require(len(runs) == 4 and all(r['status'] == 'success' for r in runs), 'Four completed runs')
    for mode in ('off', 'on'):
        selected = [r for r in runs if r['job']['enabled'] == (mode == 'on')]
        summary = evidence['summary']['metrics_by_mode']['P0_' + mode]
        require(len(selected) == summary['n'] == 2, f'{mode}: sample count')
        require(abs(statistics.median(r['sampler']['denoise_seconds'] for r in selected) - summary['sampling_median_seconds']) < 1e-8, f'{mode}: sampling median')
        require(abs(statistics.median(r['workflow_seconds'] for r in selected) - summary['workflow_median_seconds']) < 1e-8, f'{mode}: workflow median')
        require(abs(statistics.median(r['gpu_sampling_peak_bytes'] / 2**30 for r in selected) - summary['nvml_sampling_median_gib']) < 1e-8, f'{mode}: NVML sampling peak median')
        require(abs(statistics.median(r['sampler']['torch_peak_allocated_bytes'] / 2**30 for r in selected) - summary['torch_allocated_median_gib']) < 1e-8, f'{mode}: Torch allocated median')
    with (root / 'validation/4k28_runs.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 4, 'CSV completed run count')
    for row, run in zip(rows, runs):
        require(row['run_id'] == run['run_id'] and float(row['sampling_seconds']) == run['sampler']['denoise_seconds'], 'CSV timing/source equivalence')
        require(row['latent_sha256'] == run['sampler']['latent_sha256'] and row['pixel_sha256'] == run['pixel_sha256'], 'CSV output hashes')
        require(row['mode'] == ('ON' if run['job']['enabled'] else 'OFF'), 'CSV mode')
        for field in ('width', 'height', 'steps', 'seed', 'area_seed'):
            require(int(row[field]) == run['job'][field], f'CSV {field}')
        for field, expected in (
            ('workflow_seconds', run['workflow_seconds']),
            ('nvml_sampling_peak_gib', run['gpu_sampling_peak_bytes'] / 2**30),
            ('nvml_whole_workflow_peak_gib', run['gpu_whole_workflow_peak_bytes'] / 2**30),
            ('torch_allocated_peak_gib', run['sampler']['torch_peak_allocated_bytes'] / 2**30),
        ):
            require(abs(float(row[field]) - expected) < 1e-8, f'CSV {field}')
    checked.append('Recorded sample counts, medians and per-run CSV agree with historical JSON')
    require((root / 'VERSION.txt').read_text().strip() == baseline['prerelease_version'], 'Consistent prerelease version')
    if check_manifest:
        manifest = read(root / 'MANIFEST.json')
        expected = manifest['files']
        actual = {p.relative_to(root).as_posix(): sha(p) for p in paths if p.name != 'MANIFEST.json'}
        require(actual == expected, 'Manifest file list and SHA256 values')
        checked.append('Manifest complete file list and SHA256 values')
    return {'status': 'passed', 'version': baseline['prerelease_version'], 'checks': checked,
            'annotated_ui_workflows': 6, 'matching_api_prompts': 6,
            'default_node_names_preserved': True, 'gpu_generation_performed': False,
            'network_or_server_requests_performed': False}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-manifest', action='store_true')
    parser.add_argument('--report', type=Path, help='Optional JSON report, relative to repository root')
    args = parser.parse_args()
    result = verify(check_manifest=args.check_manifest)
    if args.report:
        output = (ROOT / args.report).resolve()
        require(output.is_relative_to(ROOT), 'Report must stay within the repository')
        output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
