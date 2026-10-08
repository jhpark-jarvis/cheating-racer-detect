from pathlib import Path
import tempfile
import unittest

from cheating_racer_detect.tools import Toolchain
from scripts.benchmark_events import DiskSampler, classify, own_memory


class Baseline(unittest.TestCase):
    def test_classification_and_sampled_owned_bytes(self):
        tools = Toolchain(Path('ffmpeg'), Path('ffprobe'))
        source = Path('fixture.mp4')
        self.assertEqual(classify([tools.ffprobe, '-show_frames', source], source, tools), 'source_timeline_probe')
        self.assertEqual(classify([tools.ffmpeg, '-i', source, '-f', 'null'], source, tools), 'source_full_decode')
        self.assertEqual(classify([tools.ffmpeg, '-version'], source, tools), 'capabilities')
        with tempfile.TemporaryDirectory(prefix='crd-benchmark-unit-') as directory:
            root = Path(directory)
            (root/'fixture').write_bytes(b'1234')
            (root/'job.partial').mkdir()
            (root/'job.partial'/'raw').write_bytes(b'123456')
            with DiskSampler(root) as sampler:
                sampler.sample()
            self.assertEqual(sampler.peak, 10)
            self.assertEqual(sampler.partial_peak, 6)
            self.assertGreaterEqual(sampler.samples, 1)

    def test_own_process_memory_units(self):
        memory = own_memory()
        if memory is not None:
            self.assertGreater(memory['working_set_bytes'], 0)
            self.assertGreaterEqual(memory['peak_working_set_bytes_since_process_start'], memory['working_set_bytes'])


if __name__ == '__main__':
    unittest.main()
