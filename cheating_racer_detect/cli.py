"""Command line entry point. Expected errors are JSON and never contain tool stderr."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

from . import __version__
from .errors import AppError
from .service import create_clip
from .tools import discover_tools, doctor


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise AppError('INVALID_ARGUMENT', '명령 인자를 확인하세요. --help로 사용법을 볼 수 있습니다.')


def main(argv: list[str] | None = None) -> int:
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = Parser(description='로컬 블랙박스 검토용 클립 생성')
    parser.add_argument('--version', action='version', version=__version__)
    commands = parser.add_subparsers(dest='command', required=True)
    doctor_parser = commands.add_parser('doctor', help='FFmpeg 도구·코덱 확인')
    doctor_parser.add_argument('--ffmpeg-dir', type=Path)
    clip_parser = commands.add_parser('clip', help='원본을 보존하면서 지정 구간 추출')
    clip_parser.add_argument('--input', required=True)
    clip_parser.add_argument('--start', required=True, help='비디오 시작 기준 초')
    clip_parser.add_argument('--end', required=True, help='포함하지 않을 끝 시각(초)')
    clip_parser.add_argument('--output', required=True, help='아직 존재하지 않는 결과 디렉터리')
    clip_parser.add_argument('--ffmpeg-dir', type=Path)
    try:
        args = parser.parse_args(argv)
        if args.command == 'doctor':
            result = doctor(discover_tools(args.ffmpeg_dir))
        else:
            try:
                start, end = Decimal(args.start), Decimal(args.end)
            except InvalidOperation:
                raise AppError('INVALID_RANGE', '시작과 끝은 초 단위 숫자여야 합니다.') from None
            # Validate media paths before discovering or executing media tools.
            tools = discover_tools(args.ffmpeg_dir) if args.ffmpeg_dir else None
            create_clip(args.input, start, end, args.output, tools)
            result = {'status': 'complete', 'output': str(Path(args.output).absolute()),
                      'clip': 'clip.mp4', 'metadata': 'result.json'}
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except KeyboardInterrupt:
        error = AppError('CANCELLED', '작업이 취소되었습니다. 같은 요청을 다시 실행할 수 있습니다.')
        code = 130
    except AppError as exc:
        error = exc
        code = 2
    print(json.dumps({'status': 'error', 'code': error.code, 'message': error.message}, ensure_ascii=False), file=sys.stderr)
    return code
