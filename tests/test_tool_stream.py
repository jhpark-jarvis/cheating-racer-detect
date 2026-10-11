"""Bounded stdout and real owned child cleanup, including parser cancellation."""

import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from cheating_racer_detect.errors import AppError
from cheating_racer_detect.tools import run_tool_lines


class ToolStreamTests(unittest.TestCase):
    def test_actual_incremental_lines_and_large_discarded_stderr(self):
        lines = []
        code = 'import sys; sys.stderr.write("PRIVATE_MARKER"*100000); print("first",flush=True); print("second",flush=True)'
        run_tool_lines([sys.executable, '-c', code], lines.append)
        self.assertEqual([line.strip() for line in lines], ['first', 'second'])

    def test_actual_consumer_failure_and_cancel_kill_and_reap_child(self):
        actual_popen = subprocess.Popen
        for failure in (ValueError('fixture'), KeyboardInterrupt()):
            children = []
            def spawn(*args, **kwargs):
                process = actual_popen(*args, **kwargs)
                children.append(process)
                return process
            def consume(line):
                raise failure
            with self.subTest(failure=type(failure).__name__), \
                    patch('cheating_racer_detect.tools.subprocess.Popen', side_effect=spawn), \
                    self.assertRaises(type(failure)):
                run_tool_lines([sys.executable, '-c', 'import time; print("ready",flush=True); time.sleep(30)'], consume)
            self.assertEqual(len(children), 1)
            self.assertIsNotNone(children[0].poll())
            self.assertTrue(children[0].stdout.closed)
        run_tool_lines([sys.executable, '-c', 'print("retry")'], lambda line: None)

    def test_actual_long_unterminated_invalid_utf8_and_nonzero_are_rejected(self):
        programs = ['import sys; sys.stdout.write("x"*5000)',
                    'import sys; sys.stdout.write("no-newline")',
                    'import sys; sys.stdout.buffer.write(b"\\xff\\n")',
                    'import sys; print("valid",flush=True); sys.stderr.write("PRIVATE_MARKER"); sys.exit(7)']
        for code in programs:
            with self.subTest(code=code), self.assertRaises(AppError) as ctx:
                run_tool_lines([sys.executable, '-c', code], lambda line: None, error_code='OUTPUT_INVALID')
            self.assertEqual(ctx.exception.code, 'OUTPUT_INVALID')
            self.assertNotIn('PRIVATE_MARKER', str(ctx.exception))

    def test_missing_binary_and_invalid_protocol_bound(self):
        with tempfile.TemporaryDirectory(prefix='crd-stream-tool-') as root, self.assertRaises(AppError) as ctx:
            run_tool_lines([Path(root)/'absent-tool'], lambda line: None)
        self.assertEqual(ctx.exception.code, 'DEPENDENCY_MISSING')
        for bound in (0, -1, 1.5, True):
            with self.subTest(bound=bound), self.assertRaises(ValueError), \
                    patch('cheating_racer_detect.tools.subprocess.Popen') as spawn:
                run_tool_lines(['fixture'], lambda line: None, max_line_bytes=bound)
            spawn.assert_not_called()

    def test_io_failure_closes_pipe_and_does_not_buffer_during_cleanup(self):
        class BadPipe(io.BytesIO):
            def readline(self, size):
                self.bound = size
                raise OSError('PRIVATE_MARKER')
        class Child:
            def __init__(self):
                self.stdout = BadPipe()
                self.returncode = None
                self.killed = False
            def poll(self):
                return self.returncode
            def kill(self):
                self.killed = True
                self.returncode = -9
            def wait(self):
                return self.returncode
        child = Child()
        with patch('cheating_racer_detect.tools.subprocess.Popen', return_value=child) as spawn, \
                self.assertRaises(AppError) as ctx:
            run_tool_lines(['fixture', 'literal;not-a-shell'], lambda line: None, max_line_bytes=128)
        self.assertEqual(child.stdout.bound, 129)
        self.assertTrue(child.killed)
        self.assertTrue(child.stdout.closed)
        self.assertEqual(spawn.call_args.kwargs['stderr'], subprocess.DEVNULL)
        self.assertFalse(spawn.call_args.kwargs['shell'])
        self.assertNotIn('PRIVATE_MARKER', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
