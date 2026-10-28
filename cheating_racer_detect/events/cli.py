"""Installed experimental event commands; no model loading or downloads."""

from pathlib import Path

from ..errors import AppError
from ..tools import discover_tools


def register(commands):
    events = commands.add_parser('events', help='기록된 실험 입력의 목록·재생·검증·평가')
    actions = events.add_subparsers(dest='event_command', required=True)
    listing = actions.add_parser('list', help='원본과 관측 입력을 검증해 후보 목록 저장')
    listing.add_argument('--lane-diagnostics', action='store_true', help='차선 출처·unknown·단절 진단 함께 저장/표시')
    listing.add_argument('--lamp-diagnostics', action='store_true', help='좌우 램프 선언·ROI·공백·단절 진단 함께 저장/표시')
    replay = actions.add_parser('replay', help='명시 선택 후보의 검토 묶음 생성')
    for parser in (listing, replay):
        parser.add_argument('--input', required=True, type=Path, help='원본 로컬 MP4')
        parser.add_argument('--inputs', required=True, type=Path, help='기록된 inputs.json')
        parser.add_argument('--output', required=True, type=Path, help='아직 존재하지 않는 결과 폴더')
        parser.add_argument('--ffmpeg-dir', type=Path)
    replay.add_argument('--catalog', type=Path, help='선택적 catalog.json; 일괄 선택에만 사용')
    selection = replay.add_mutually_exclusive_group(required=True)
    selection.add_argument('--candidate-index', type=int, help='0-based 단일 후보 번호')
    selection.add_argument('--candidate-indices', nargs='+', type=int, help='중복 없는 일괄 후보 번호·순서')
    verify = actions.add_parser('verify', help='저장 batch 읽기 전용 검사; 선택적 원본 재생')
    verify.add_argument('--bundle', required=True, type=Path)
    verify.add_argument('--input', type=Path, help='선택적 원본 MP4')
    verify.add_argument('--ffmpeg-dir', type=Path)
    listing_check = actions.add_parser('verify-list', help='저장 후보 목록·선택 진단 읽기 전용 검사')
    listing_check.add_argument('--listing', required=True, type=Path)
    listing_check.add_argument('--input', type=Path, help='선택적 원본 MP4; 새 재생 대조')
    listing_check.add_argument('--ffmpeg-dir', type=Path)
    listing_check.add_argument('--require-diagnostics', nargs='+', choices=('lane', 'lamp'), default=[])
    assess = actions.add_parser('evaluate', help='명시적인 실험 정답·예측의 사건/신호 평가')
    assess.add_argument('--labels', required=True, type=Path)


def review_dependencies():
    """Only replay rendering needs these; basic CLI/list/verify/evaluation do not."""
    try:
        import numpy  # noqa: F401
        import cv2  # noqa: F401
    except (ImportError, OSError):
        raise AppError('DEPENDENCY_MISSING', '후보 재생에는 tracking 선택 의존성 NumPy/OpenCV가 필요합니다.') from None


def run(args):
    try:
        return _run(args)
    except (TypeError, ValueError, KeyError, AttributeError, OverflowError, ZeroDivisionError, RecursionError, UnicodeError):
        code = 'INVALID_ASSESSMENT' if args.event_command == 'evaluate' else 'INVALID_INPUTS'
        raise AppError(code, '실험 입력의 형식·시각·선택 조건을 확인하세요.') from None
    except OSError:
        raise AppError('IO_ERROR', '로컬 파일 접근 권한과 저장 공간을 확인하세요.') from None


def _run(args):
    if args.event_command == 'verify-list':
        from .listing import verify_listing
        if args.ffmpeg_dir is not None and args.input is None:
            raise AppError('INVALID_ARGUMENT', '--ffmpeg-dir는 원본 재생에만 사용하세요.')
        tools = discover_tools(args.ffmpeg_dir) if args.ffmpeg_dir is not None else None
        return verify_listing(args.listing, source=args.input, toolchain=tools,
                              required_diagnostics=args.require_diagnostics)
    if args.event_command == 'evaluate':
        from .assessment import from_report
        from .verify import Files
        from ..paths import local_path
        path = local_path(args.labels)
        files = Files(path.parent)
        result = from_report(files.json(path.name))
        files.finish()
        return result
    if args.event_command == 'verify':
        from .verify import verify_batch
        tools = discover_tools(args.ffmpeg_dir) if args.ffmpeg_dir is not None else None
        return verify_batch(args.bundle, source=args.input, toolchain=tools)
    if args.event_command == 'replay':
        if args.catalog is not None and args.candidate_indices is None:
            raise AppError('INVALID_ARGUMENT', '--catalog는 --candidate-indices와 함께 사용하세요.')
        review_dependencies()
    from .inputs import load_inputs
    inputs = load_inputs(args.inputs)
    tools = discover_tools(args.ffmpeg_dir) if args.ffmpeg_dir is not None else None
    if args.event_command == 'list':
        from .catalog import export_catalog, export_catalog_with_diagnostics, listing_summary
        if args.lamp_diagnostics:
            from .catalog import export_catalog_with_lamp_diagnostics
            catalog, lamps, lanes = export_catalog_with_lamp_diagnostics(
                args.input, inputs, args.output, toolchain=tools, include_lane_diagnostics=args.lane_diagnostics)
            return listing_summary(catalog, lanes, lamps)
        if args.lane_diagnostics:
            catalog, diagnostics = export_catalog_with_diagnostics(args.input, inputs, args.output, toolchain=tools)
            return listing_summary(catalog, diagnostics)
        return listing_summary(export_catalog(args.input, inputs, args.output, toolchain=tools))
    from .replay import export_replay, export_replay_batch
    if args.candidate_indices is None:
        result = export_replay(args.input, inputs, args.output, candidate_index=args.candidate_index, toolchain=tools)
        summary = {key: result[key] for key in ('status', 'candidate_index', 'signal_state')}
    else:
        from .catalog import load_catalog
        kwargs = {} if args.catalog is None else {'catalog': load_catalog(args.catalog)}
        result = export_replay_batch(args.input, inputs, args.output, candidate_indices=args.candidate_indices, toolchain=tools, **kwargs)
        summary = {key: result[key] for key in ('status', 'candidate_indices', 'selected_count')}
    return {**summary, 'scope': 'recorded inputs; not automatic recognition'}
