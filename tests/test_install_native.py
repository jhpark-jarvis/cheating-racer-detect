"""Real backend/archive/clean install, not editable or checkout imports."""

import importlib.metadata
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import unittest

from cheating_racer_detect.errors import AppError
from cheating_racer_detect.tools import discover_tools
from scripts.check_install import run

try:
    TOOLS = discover_tools()
except AppError:
    TOOLS = None
UV = shutil.which('uv')
AVAILABLE = all(importlib.util.find_spec(n) is not None for n in ('numpy', 'cv2', 'setuptools'))
if AVAILABLE:
    AVAILABLE = int(importlib.metadata.version('setuptools').split('.')[0]) >= 77


@unittest.skipUnless(os.name == 'nt' and TOOLS and UV and AVAILABLE,
                     'NOT_RUN: Windows/existing FFmpeg/uv/setuptools>=77/NumPy/OpenCV required')
class Installed(unittest.TestCase):
    def test_actual_archives_rebuild_clean_base_install_and_external_commands(self):
        result = run(Path(UV), Path(sys.executable), TOOLS.ffmpeg.parent)
        self.assertEqual(result['status'], 'PASS')
        self.assertTrue(result['sdist_rebuild'])
        self.assertEqual(result['wheel']['base_dependencies'], 0)
        self.assertTrue(result['noneditable_external_cwd'])
        self.assertFalse(result['base_ml_packages'])
        self.assertEqual(result['base_list_candidates'], 2)
        self.assertTrue(result['base_verify_source'])
        self.assertTrue(result['base_lane_contract'])
        self.assertTrue(result['base_lane_section'])
        self.assertEqual(result['dependency_error'], 'DEPENDENCY_MISSING')
        self.assertEqual(result['optional']['status'], 'NOT_RUN')
        self.assertTrue(result['temporary_cleanup'])


if __name__ == '__main__':
    unittest.main()
