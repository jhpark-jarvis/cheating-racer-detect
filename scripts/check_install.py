"""Offline wheel/sdist + clean Windows install smoke. No package publication.

Use an existing Python with setuptools>=77, existing uv and FFmpeg. Optional
tracking installation uses an existing uv cache with --offline, never downloads.
The smoke host needs NumPy/OpenCV to prepare a synthetic review bundle; the
clean base installation under test deliberately has neither package.
All generated distributions, venvs, fixture media and outputs are temporary.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import venv

from cheating_racer_detect.paths import local_path
from cheating_racer_detect.tools import discover_tools
from .package_artifacts import check_sdist, check_wheel, require, snapshot, sources

ROOT = Path(__file__).resolve().parents[1]


def environment():
    result = os.environ.copy()
    for name in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'UV_PROJECT_ENVIRONMENT', 'CRD_FFMPEG_DIR'):
        result.pop(name, None)
    result.update(PYTHONNOUSERSITE='1', PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1', UV_OFFLINE='1')
    return result


def child(command, cwd, *, expected=0, parsed=False):
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    process = subprocess.Popen([str(s) for s in command], cwd=cwd, env=environment(), stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags)
    try:
        out, err = process.communicate(timeout=120)
        if process.returncode != expected:
            raise RuntimeError('Install smoke subprocess failed: '+str(process.returncode))
        if parsed:
            payload = out if expected == 0 else err
            return json.loads(payload.decode('utf-8'))
        return out.decode('utf-8', errors='replace')
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()


def build(python, root, destination):
    destination.mkdir()
    child([python, '-c', 'import setuptools.build_meta as b,sys; d=sys.argv[1]; b.build_sdist(d); b.build_wheel(d)', destination], root)
    wheels, archives = list(destination.glob('*.whl')), list(destination.glob('*.tar.gz'))
    require(len(wheels) == len(archives) == 1)
    return wheels[0], archives[0]


def install(uv, python, wheel, cwd):
    child([uv, 'pip', 'install', '--no-index', '--no-deps', '--no-cache', '--python', python, wheel], cwd)


def run(uv, build_python, ffmpeg_dir, cache=None):
    if os.name != 'nt':
        raise RuntimeError('Install smoke validated on Windows only')
    uv, build_python = local_path(uv), local_path(build_python)
    require(uv.is_file() and build_python.is_file())
    tools = discover_tools(ffmpeg_dir)
    # Synthetic caller receipts are explicitly not automatic model detections.
    from .demo_batch_replay import generate_inputs
    from .demo_assessment import definition
    from cheating_racer_detect.events.inputs import save_inputs
    from cheating_racer_detect.service import _sha256
    from cheating_racer_detect.events.replay import export_replay_batch
    source_bytes = sources(ROOT)
    with tempfile.TemporaryDirectory(prefix='crd-installed-') as directory:
        temp = Path(directory)
        copied = temp/'source'
        snapshot(ROOT, copied)
        # Build real artifacts from a staged tree with accidental-file sentinels.
        for name in ('AGENTS.md', 'docs/private.md', '.tools/secret.pth', 'data/footage.mp4',
                     'cheating_racer_detect/local-secret.txt', 'tests/test_private.py'):
            path = copied/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('NOT_FOR_DISTRIBUTION', encoding='utf-8')
        wheel, archive = build(build_python, copied, temp/'dist')
        wheel_audit, sdist_audit = check_wheel(wheel, ROOT), check_sdist(archive, ROOT)
        unpack = temp/'unpacked'
        unpack.mkdir()
        with tarfile.open(archive) as stream:
            stream.extractall(unpack, filter='data')
        rebuilt, _again = build(build_python, unpack/sdist_audit['prefix'], temp/'rebuilt')
        require(check_wheel(rebuilt, ROOT) == wheel_audit)
        external = temp/'outside-source'
        external.mkdir()
        clean = temp/'venv'
        venv.EnvBuilder(with_pip=False, system_site_packages=False).create(clean)
        python, executable = clean/'Scripts/python.exe', clean/'Scripts/crd.exe'
        install(uv, python, rebuilt, external)
        imported = child([python, '-I', '-c', 'import cheating_racer_detect as c,importlib.util as u,json; '
                          'print(json.dumps({"path":c.__file__,"version":c.__version__,"native":any(u.find_spec(n) for n in ("numpy","cv2","torch"))}))'], external, parsed=True)
        require(Path(imported['path']).is_relative_to(clean) and imported['native'] is False)
        lane_contract = child([python, '-I', '-c',
            'from cheating_racer_detect.analysis import LaneBatch,LaneSlot; '
            'from cheating_racer_detect.tracking import Frame; from fractions import Fraction; '
            'import json; f=Frame("fixture",0,0,500,Fraction(1,10),500,96,64); '
            's=tuple(LaneSlot(i,a,(False,)*n,(None,)*n) for i,a,n in '
            '((1,"row",72),(2,"row",72),(0,"column",81),(3,"column",81))); '
            'print(json.dumps(LaneBatch(f,"fixture",0,s).report()))'], external, parsed=True)
        require(lane_contract['frame']['pts'] == 500 and len(lane_contract['slots']) == 4)
        section_contract = child([python, '-I', '-c',
            'from cheating_racer_detect.analysis import LaneBatch,LaneSlot; '
            'from cheating_racer_detect.events import BoundaryBinding,LaneContext,LaneSectionBridge,'
            'LaneConfig,Vehicle,section_for_vehicle; '
            'from cheating_racer_detect.tracking import Frame,Box,Detection; '
            'from fractions import Fraction as F; import json; '
            'f=Frame("fixture",0,0,500,F(1,10),500,96,64); '
            'v=Vehicle(f,(1,2,1),Detection(0,2,Box(20,10,30,30),.9,"fixture",f)); '
            's=tuple(LaneSlot(i,a,(True,)*n,tuple((x,63.*j/(n-1)) for j in range(n))) '
            'for i,a,n,x in ((1,"row",72,10.),(2,"row",72,48.),'
            '(0,"column",81,86.),(3,"column",81,94.))); '
            'b=LaneBatch(f,"lanes-fixture",0,s); '
            'bindings=tuple(BoundaryBinding(i,k,(True,)*n) for i,k,n in '
            '((1,"left",71),(2,"middle",71),(0,"right",80))); '
            'c=LaneContext(f,v.identity,b.model_id,"fixture-context","road",bindings,'
            'True,True,"stable","ordinary",True); '
            'r=section_for_vehicle(f,v,b,c); '
            'bridge=LaneSectionBridge(LaneConfig("fixture-policy",.15,F(1,5),F(3,5),10.)); '
            'assert bridge.update(f,v,b,c)["lane"]["status"]=="warming"; '
            'print(json.dumps(r.report()))'], external, parsed=True)
        require(section_contract['section']['x'] == [10.,48.,86.] and
                section_contract['status'] == 'ready' and section_contract['frame']['pts'] == 500)
        require(child([executable, '--version'], external).strip() == imported['version'])
        for action in ('list', 'replay', 'verify', 'evaluate'):
            child([executable, 'events', action, '--help'], external)
        child([python, '-I', '-m', 'cheating_racer_detect', 'events', '--help'], external)
        missing_tools = child([executable, 'doctor', '--ffmpeg-dir', temp/'no-tools'], external, expected=2, parsed=True)
        require(missing_tools['code'] == 'DEPENDENCY_MISSING')
        child([executable, 'doctor', '--ffmpeg-dir', tools.ffmpeg.parent], external, parsed=True)
        fixture = temp/'fixture'
        fixture.mkdir()
        source, inputs = generate_inputs(tools, fixture)
        original_hash = _sha256(source)
        # Keep legacy execution coverage, then exercise actual installed CLI
        # with the new explicit synthetic v2 evidence (including offline audit).
        save_inputs(inputs, fixture/'legacy')
        legacy_listing = child([executable, 'events', 'list', '--input', source,
            '--inputs', fixture/'legacy/inputs.json', '--output', fixture/'legacy-list',
            '--ffmpeg-dir', tools.ffmpeg.parent], external, parsed=True)
        require(legacy_listing['candidate_count'] == 2)
        from .demo_lane_replay import with_lane_evidence
        inputs = with_lane_evidence(inputs)
        save_inputs(inputs, fixture/'recorded')
        record = fixture/'recorded/inputs.json'
        child([executable, 'clip', '--input', source, '--start', '0.35', '--end', '0.85',
               '--output', fixture/'clip', '--ffmpeg-dir', tools.ffmpeg.parent], external, parsed=True)
        listing = child([executable, 'events', 'list', '--input', source, '--inputs', record,
                         '--output', fixture/'listed', '--ffmpeg-dir', tools.ffmpeg.parent], external, parsed=True)
        require(listing['candidate_count'] == 2)
        labels = fixture/'labels.json'
        labels.write_text(json.dumps(definition()), encoding='utf-8')
        assessment = child([executable, 'events', 'evaluate', '--labels', labels], external, parsed=True)
        require(assessment['events']['fn'] == 1 and assessment['end_to_end_known_prediction_coverage'] == '1/2')
        replay_args = [executable, 'events', 'replay', '--input', source, '--inputs', record,
                       '--candidate-indices', '1', '0', '--catalog', fixture/'listed/catalog.json',
                       '--output', fixture/'batch', '--ffmpeg-dir', tools.ffmpeg.parent]
        missing = child(replay_args, external, expected=2, parsed=True)
        require(missing['code'] == 'DEPENDENCY_MISSING' and not (fixture/'batch').exists())
        # A prepared bundle proves that read-only source verification also works
        # in the actual no-ML installed environment, independent of this helper.
        export_replay_batch(source, inputs, fixture/'prepared', candidate_indices=(0,), toolchain=tools)
        verified = child([executable, 'events', 'verify', '--bundle', fixture/'prepared', '--input', source,
                          '--ffmpeg-dir', tools.ffmpeg.parent], external, parsed=True)
        require(verified['status'] == 'verified' and verified['source_replayed'])
        verified_offline = child([executable, 'events', 'verify', '--bundle', fixture/'prepared',
                                 '--ffmpeg-dir', tools.ffmpeg.parent], external, parsed=True)
        require(verified_offline['status'] == 'verified')
        child([executable, 'events', 'evaluate', '--labels', fixture/'missing'], external, expected=2, parsed=True)
        optional = {'status': 'NOT_RUN', 'reason': 'No existing offline cache requested'}
        if cache is not None:
            cache = local_path(cache)
            child([uv, 'pip', 'install', '--offline', '--cache-dir', cache, '--link-mode', 'copy', '--python', python,
                   'numpy==2.2.6', 'opencv-python==4.11.0.86'], external)
            child(replay_args, external, parsed=True)
            child([executable, 'events', 'verify', '--bundle', fixture/'batch', '--input', source,
                   '--ffmpeg-dir', tools.ffmpeg.parent], external, parsed=True)
            before = (fixture/'batch/batch.json').read_bytes()
            exists = child(replay_args, external, expected=2, parsed=True)
            require(exists['code'] == 'OUTPUT_EXISTS' and (fixture/'batch/batch.json').read_bytes() == before)
            require((fixture/'batch/catalog.json').read_bytes() == (fixture/'listed/catalog.json').read_bytes())
            lane_decoded = child([python, '-I', '-c',
                'import numpy as n,json; from fractions import Fraction; '
                'from cheating_racer_detect.tracking import Frame; '
                'from cheating_racer_detect.analysis import UFLDOutput,decode_ufld; '
                'f=Frame("fixture",0,0,500,Fraction(1,10),500,96,64); '
                'p={k:n.zeros(s,dtype=n.float32) for k,s in '
                '{"loc_row":(1,200,72,4),"loc_col":(1,100,81,4),'
                '"exist_row":(1,2,72,4),"exist_col":(1,2,81,4)}.items()}; '
                'print(json.dumps(decode_ufld(f,UFLDOutput(f,"fixture",0,p)).report()))'], external, parsed=True)
            require(lane_decoded['frame']['pts'] == 500 and
                    all(not slot['support_passed'] for slot in lane_decoded['slots']))
            optional = {'status': 'PASS', 'selected_order': [1, 0], 'catalog_checked': True,
                        'existing_preserved': True, 'lane_decode': True}
        require(_sha256(source) == original_hash and sources(ROOT) == source_bytes)
        result = {'status': 'PASS', 'wheel': wheel_audit, 'sdist': sdist_audit,
                  'sdist_rebuild': True, 'noneditable_external_cwd': True, 'base_ml_packages': False,
                  'base_list_candidates': 2, 'base_verify_source': True, 'dependency_error': missing['code'],
                  'base_lane_contract': True,
                  'base_lane_section': True,
                  'base_lane_provenance': True,
                  'optional': optional, 'source_preserved': True}
    return {**result, 'temporary_cleanup': not temp.exists()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--uv', required=True, type=Path)
    parser.add_argument('--build-python', required=True, type=Path)
    parser.add_argument('--ffmpeg-dir', required=True, type=Path)
    parser.add_argument('--tracking-cache', type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.uv, args.build_python, args.ffmpeg_dir, args.tracking_cache)))
        return 0
    except KeyboardInterrupt:
        print(json.dumps({'status': 'cancelled', 'code': 'CANCELLED'}))
        return 130
    except Exception as error:
        print(json.dumps({'status': 'error', 'code': type(error).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
