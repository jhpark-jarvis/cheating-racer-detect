"""Real FFmpeg F01--F07 integration, independent decoded content and PTS oracle.

Run: python -m unittest tests.test_integration -v
All synthetic media is made in an owned OS temporary directory and removed in
class cleanup. A missing media tool is reported as SKIPPED / NOT_RUN, never PASS.
No product detector, GPU, model, package download or real driving data is used.
"""

from __future__ import annotations

from decimal import Decimal
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from tests.fixtures import (
    COUNT, FPS, HEIGHT, SAMPLE_RATE, VFR_IDS, WIDTH, frame_error, generate,
    pcm_samples, probe, rgb_frames, run, sha256, timestamps,
)


class RealMediaIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from cheating_racer_detect.errors import AppError
        from cheating_racer_detect.tools import discover_tools

        try:
            cls.tools = discover_tools()
        except AppError as error:
            if error.code == "DEPENDENCY_MISSING":
                raise unittest.SkipTest("NOT_RUN: real FFmpeg / ffprobe unavailable") from error
            raise
        cls.temporary = tempfile.TemporaryDirectory(prefix="crd-integration-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.media = generate(cls.tools, cls.root / "합성 영상 자료")

    def clip(self, fixture: str, start: str, end: str, name: str | None = None):
        from cheating_racer_detect.service import create_clip

        output = self.root / (name or self.id().rsplit(".", 1)[-1])
        result = create_clip(
            self.media[fixture], Decimal(start), Decimal(end), output,
            toolchain=self.tools,
        )
        self.assertEqual(set(path.name for path in output.iterdir()), {"clip.mp4", "result.json"})
        persisted = json.loads((output / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted, result)
        def strings(value):
            if isinstance(value, str):
                yield value
            elif isinstance(value, dict):
                for item in value.values():
                    yield from strings(item)
            elif isinstance(value, list):
                for item in value:
                    yield from strings(item)

        for value in strings(persisted):
            self.assertNotIn(str(self.root), value)
            self.assertNotIn(self.root.as_posix(), value)
        self.assertEqual(persisted["schema_version"], 1)
        self.assertEqual(persisted["status"], "complete")
        self.assertEqual(persisted["source"]["sha256"], sha256(self.media[fixture]))
        self.assertEqual(persisted["source"]["size_bytes"], self.media[fixture].stat().st_size)
        source_pts = timestamps(self.tools, self.media[fixture])
        normalized = [Decimal(str(round(value - source_pts[0], 6))) for value in source_pts]
        included = [value for value in normalized if Decimal(start) <= value < Decimal(end)]
        mapping = persisted["media"]["selection"]
        self.assertEqual(Decimal(mapping["first_normalized_seconds"]), included[0])
        self.assertEqual(Decimal(mapping["last_normalized_seconds"]), included[-1])
        self.assertEqual(mapping["frame_count"], len(included))
        return output, result

    def assert_content_and_timing(self, source: Path, output: Path, start: float, end: float):
        source_pts = timestamps(self.tools, source)
        source_frames = rgb_frames(self.tools, source)
        self.assertEqual(len(source_pts), len(source_frames))
        relative = [round(value - source_pts[0], 6) for value in source_pts]
        selected = [index for index, value in enumerate(relative) if start <= value < end]
        self.assertTrue(selected, "Test interval must contain at least one frame")
        output_pts = timestamps(self.tools, output)
        output_frames = rgb_frames(self.tools, output)
        self.assertEqual(len(output_frames), len(selected), "No selected frame may be lost or duplicated")
        self.assertEqual(len(output_pts), len(selected))
        self.assertAlmostEqual(output_pts[0], relative[selected[0]] - start, delta=0.001)
        for output_index, source_index in enumerate(selected):
            with self.subTest(frame=output_index, original_frame=source_index):
                # Mean absolute error below 8 intensity units leaves room for
                # H.264 re-encoding; nearest distinct source frame must match too.
                errors = [frame_error(output_frames[output_index], frame) for frame in source_frames]
                self.assertEqual(min(range(len(errors)), key=errors.__getitem__), source_index)
                self.assertLess(errors[source_index], 8.0)
                self.assertAlmostEqual(
                    output_pts[output_index] - output_pts[0],
                    source_pts[source_index] - source_pts[selected[0]], delta=0.001,
                )
        return relative[selected[0]], relative[selected[-1]]

    def assert_failure(self, source: Path, start: str, end: str, code: str, name: str):
        from cheating_racer_detect.errors import AppError
        from cheating_racer_detect.service import create_clip

        output = self.root / name
        original_hash = sha256(source) if source.exists() and source.is_file() else None
        before = set(self.root.iterdir())
        with self.assertRaises(AppError) as context:
            create_clip(source, Decimal(start), Decimal(end), output, toolchain=self.tools)
        self.assertEqual(context.exception.code, code)
        self.assertEqual(set(self.root.iterdir()), before, "Failure must clean its owned temporary directory")
        self.assertFalse(output.exists())
        if original_hash:
            self.assertEqual(sha256(source), original_hash)

    def test_f01_cfr_nonkeyframe_content_and_source_unchanged(self):
        source = self.media["cfr"]
        digest = sha256(source)
        # Frame 4 is not a keyframe: this catches fast stream-copy seeking.
        frames = probe(self.tools, source, frames=True)["frames"]
        self.assertEqual(len(frames), COUNT)
        self.assertEqual(frames[4]["key_frame"], 0)
        output, _ = self.clip("cfr", "0.35", "2.35")
        self.assertEqual(self.assert_content_and_timing(source, output / "clip.mp4", 0.35, 2.35), (0.4, 2.3))
        self.assertEqual(sha256(source), digest)

    def test_f02_vfr_selected_frames_and_original_spacing(self):
        source = self.media["vfr"]
        pts = timestamps(self.tools, source)
        self.assertEqual([round(value * FPS) for value in pts], list(VFR_IDS))
        self.assertGreater(len({round(b - a, 3) for a, b in zip(pts, pts[1:])}), 1)
        output, _ = self.clip("vfr", "0.25", "2.6")
        self.assert_content_and_timing(source, output / "clip.mp4", 0.25, 2.6)

    def test_f02_nonzero_pts_normalized_to_video_start(self):
        source = self.media["offset"]
        self.assertGreater(timestamps(self.tools, source)[0], 4.9)
        output, _ = self.clip("offset", "0.35", "2.35")
        self.assert_content_and_timing(source, output / "clip.mp4", 0.35, 2.35)

    def test_f03_no_audio_does_not_gain_track(self):
        output, _ = self.clip("cfr", "0", "1")
        self.assertEqual([s["codec_type"] for s in probe(self.tools, output / "clip.mp4")["streams"]], ["video"])

    def test_f03_flash_beep_sync_with_offset_and_aac_priming(self):
        # Technical tolerance fixed before measurement: max(one CFR frame,
        # one AAC LC block) = max(0.1, 1024 / 48000) = 0.1 seconds.
        tolerance = max(1 / FPS, 1024 / SAMPLE_RATE)
        for fixture in ("audio", "offset"):
            with self.subTest(fixture=fixture):
                output, _ = self.clip(fixture, "0.35", "2.35", f"av-{fixture}")
                clip = output / "clip.mp4"
                streams = probe(self.tools, clip)["streams"]
                audio = [stream for stream in streams if stream["codec_type"] == "audio"]
                self.assertEqual(len(audio), 1)
                samples = pcm_samples(self.tools, clip)
                self.assertTrue(samples)
                amplitude = max(abs(value) for value in samples)
                self.assertGreater(amplitude, 1000)
                beep_index = next(index for index, value in enumerate(samples) if abs(value) > amplitude * 0.2)
                beep_time = float(audio[0].get("start_time", 0)) + beep_index / SAMPLE_RATE
                frames = rgb_frames(self.tools, clip)
                pts = timestamps(self.tools, clip)
                # Right-hand half is white only during the synthetic flash.
                brightness = [
                    sum(frame[(row * WIDTH + WIDTH // 2) * 3:(row * WIDTH + WIDTH) * 3])
                    for frame in frames for row in (HEIGHT // 4,)
                ]
                flash_time = pts[max(range(len(brightness)), key=brightness.__getitem__)]
                self.assertLessEqual(abs(flash_time - beep_time), tolerance)

    def test_f03_late_audio_start_retains_relative_offset(self):
        def flash_minus_beep(path):
            streams = probe(self.tools, path)["streams"]
            audio = next(stream for stream in streams if stream["codec_type"] == "audio")
            samples = pcm_samples(self.tools, path)
            threshold = max(abs(value) for value in samples) * 0.2
            onset = next(index for index, value in enumerate(samples) if abs(value) > threshold)
            beep = float(audio.get("start_time", 0)) + onset / SAMPLE_RATE
            frames = rgb_frames(self.tools, path)
            pts = timestamps(self.tools, path)
            row = HEIGHT // 4
            levels = [sum(frame[(row * WIDTH + WIDTH // 2) * 3:(row * WIDTH + WIDTH) * 3]) for frame in frames]
            flash = pts[max(range(len(levels)), key=levels.__getitem__)]
            return flash - beep

        source = self.media["audio_delayed"]
        source_delta = flash_minus_beep(source)
        self.assertAlmostEqual(source_delta, -0.5, delta=1024 / SAMPLE_RATE)
        output, _ = self.clip("audio_delayed", "0.35", "2.35")
        self.assertAlmostEqual(flash_minus_beep(output / "clip.mp4"), source_delta, delta=max(1 / FPS, 1024 / SAMPLE_RATE))

    def test_f03_no_audio_overlap_does_not_create_silent_track(self):
        output, _ = self.clip("audio_delayed", "0", "0.3")
        self.assertEqual([s["codec_type"] for s in probe(self.tools, output / "clip.mp4")["streams"]], ["video"])

    def test_f04_start_zero_and_exact_end(self):
        output, _ = self.clip("cfr", "0", "3")
        self.assert_content_and_timing(self.media["cfr"], output / "clip.mp4", 0, 3)

    def test_f04_exact_single_frame(self):
        output, _ = self.clip("cfr", "0.4", "0.5")
        self.assert_content_and_timing(self.media["cfr"], output / "clip.mp4", 0.4, 0.5)

    def test_f04_empty_interval(self):
        self.assert_failure(self.media["cfr"], "0.401", "0.499", "EMPTY_INTERVAL", "empty-result")

    def test_f04_invalid_ranges_leave_no_result(self):
        for index, (start, end) in enumerate((("-1", "1"), ("1", "1"), ("2", "1"), ("0", "3.001"))):
            with self.subTest(start=start, end=end):
                self.assert_failure(self.media["cfr"], start, end, "INVALID_RANGE", f"bad-range-{index}")

    def test_f05_rotation_preserves_display_and_no_double_rotation(self):
        for angle in (90, 180, 270):
            with self.subTest(angle=angle):
                fixture = f"rotation{angle}"
                source = self.media[fixture]
                stream = next(s for s in probe(self.tools, source)['streams'] if s['codec_type'] == 'video')
                rotations = [int(item['rotation']) % 360 for item in stream.get('side_data_list', [])
                             if 'rotation' in item]
                self.assertEqual(rotations, [angle], 'Fixture must actually carry the requested rotation')
                output, _ = self.clip(fixture, "0.35", "0.85", f"rotation-{angle}-result")
                expected = (HEIGHT, WIDTH) if angle in (90, 270) else (WIDTH, HEIGHT)
                video = next(s for s in probe(self.tools, output / 'clip.mp4')['streams'] if s['codec_type'] == 'video')
                self.assertEqual((video['width'], video['height']), expected)
                self.assertTrue(all(int(item.get('rotation', 0)) % 360 == 0
                                    for item in video.get('side_data_list', [])))
                self.assert_content_and_timing(source, output / "clip.mp4", 0.35, 0.85)

    def test_f06_existing_output_and_source_collision_are_preserved(self):
        from cheating_racer_detect.errors import AppError
        from cheating_racer_detect.service import create_clip

        existing = self.root / "already exists"
        existing.mkdir()
        sentinel = existing / "keep.txt"
        sentinel.write_text("user-owned sentinel", encoding="utf-8")
        for output in (existing, self.media["cfr"]):
            with self.subTest(output_kind="directory" if output == existing else "input"):
                digest = sha256(self.media["cfr"])
                with self.assertRaises(AppError) as context:
                    create_clip(self.media["cfr"], Decimal("0"), Decimal("1"), output, toolchain=self.tools)
                self.assertIn(context.exception.code, {"OUTPUT_EXISTS", "OUTPUT_CONFLICT"})
                self.assertEqual(sha256(self.media["cfr"]), digest)
                self.assertEqual(sentinel.read_text(encoding="utf-8"), "user-owned sentinel")

    def test_f06_cli_unicode_spaces_and_shell_characters(self):
        source = self.root / "한글 & $(echo NEVER) 영상.mp4"
        shutil.copyfile(self.media["audio"], source)
        output = self.root / "한글 & 검토 결과"
        environment = os.environ.copy()
        environment["CRD_FFMPEG_DIR"] = str(Path(self.tools.ffmpeg).parent)
        result = subprocess.run([
            sys.executable, "-m", "cheating_racer_detect", "clip", "--input", str(source),
            "--start", "0.35", "--end", "2.35", "--output", str(output),
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90, env=environment)
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))
        self.assertTrue((output / "result.json").is_file())
        self.assert_content_and_timing(source, output / "clip.mp4", 0.35, 2.35)

    def test_f07_corrupt_unsupported_and_multiple_streams(self):
        corrupt = self.root / "corrupt.mp4"
        corrupt.write_bytes(b"not an MP4 container\x00\x01")
        sources = (corrupt, self.media["unsupported"], self.media["multiple_video"], self.media["multiple_audio"])
        for index, source in enumerate(sources):
            with self.subTest(kind=index):
                self.assert_failure(source, "0", "1", "UNSUPPORTED_MEDIA", f"unsupported-{index}")

    def test_f07_bitstream_damage_outside_requested_interval_is_rejected(self):
        packets = json.loads(run([
            str(self.tools.ffprobe), "-v", "error", "-select_streams", "v:0",
            "-show_packets", "-show_entries", "packet=pos,size", "-of", "json",
            str(self.media["cfr"]),
        ]))["packets"]
        # Damage the last packet of a copy, well after requested [0.35, 0.85).
        # Container / codec metadata stays valid. A selected-region-only decode
        # would miss this, while the whole-source validation must fail.
        packet = packets[-1]
        offset, size = int(packet["pos"]), int(packet["size"])
        content = bytearray(self.media["cfr"].read_bytes())
        content[offset:offset + size] = b"\xff" * size
        corrupted = self.root / "corrupted-last-packet.mp4"
        corrupted.write_bytes(content)
        self.assert_failure(corrupted, "0.35", "0.85", "EXTRACTION_FAILED", "damaged-tail-result")

    def test_f06_missing_input(self):
        self.assert_failure(self.root / "missing.mp4", "0", "1", "INPUT_NOT_FOUND", "missing-result")

    @unittest.skipUnless(os.name == "nt", "NOT_RUN: Windows console cancellation requires Windows")
    def test_f09_real_console_cancellation_cleanup_and_retry(self):
        self.check_console_recovery(force=False)

    @unittest.skipUnless(os.name == "nt", "NOT_RUN: Windows process termination requires Windows")
    def test_f09_forced_termination_residue_and_retry(self):
        self.check_console_recovery(force=True)

    def check_console_recovery(self, *, force):
        source = self.media["audio"]
        original_hash = sha256(source)
        output = self.root / ("forced-stop-and-retry" if force else "console-cancel-and-retry")
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = subprocess.SW_HIDE
        # A private hidden console permits a real group-targeted console event
        # even when the test runner itself has no attached Windows console.
        result = subprocess.run([
            sys.executable, "-m", "tests.cancel_driver", str(source), str(output),
            str(Path(self.tools.ffmpeg).parent),
        ] + (['--force'] if force else []), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60,
            creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=startup)
        self.assertEqual(result.returncode, 0, result.stdout.decode("utf-8", errors="replace"))
        evidence = json.loads(result.stdout.decode("utf-8"))
        self.assertEqual(evidence["event"], "TerminateProcess" if force else "CTRL_BREAK_EVENT")
        if force:
            self.assertNotEqual(evidence['cancel_exit'], 0)
            self.assertTrue(evidence['crash_residue_preserved_on_retry'])
        else:
            self.assertEqual(evidence["cancel_exit"], 130)
            self.assertEqual(evidence["cancel_code"], "CANCELLED")
        self.assertTrue(evidence["observed_children"])
        self.assertTrue(evidence["children_exited"])
        self.assertTrue(evidence["temporary_cleaned"])
        self.assertEqual(evidence["retry_exit"], 0)
        self.assertEqual(sha256(source), original_hash)
        self.assert_content_and_timing(source, output / "clip.mp4", 0.35, 2.35)


if __name__ == "__main__":
    unittest.main()
