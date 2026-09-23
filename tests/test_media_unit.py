"""Exact clock and container boundary tests, independent of FFmpeg binaries."""

from fractions import Fraction
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from cheating_racer_detect.errors import AppError
from cheating_racer_detect.media import Frame, Timeline, _profile, _rotation, _timeline, select_frames, validate_container
from cheating_racer_detect.tools import Toolchain


def box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", len(payload) + 8, kind) + payload


def container(reference: bytes = b"\x00\x00\x00\x01", entry_type: bytes = b"url ") -> bytes:
    dref = box(b"dref", b"\x00" * 4 + struct.pack(">I", 1) + box(entry_type, reference))
    content = box(b"dinf", dref)
    for kind in (b"minf", b"mdia", b"trak", b"moov"):
        content = box(kind, content)
    return box(b"ftyp", b"isom\x00\x00\x00\x00isommp42") + content + box(b"mdat", b"local media")


class TimelineTests(unittest.TestCase):
    def setUp(self):
        # Nonzero source PTS and nonuniform gaps: 0, 1/10, 3/10, 7/10.
        self.timeline = Timeline(Fraction(1, 1000), tuple(Frame(p, 100) for p in (5000, 5100, 5300, 5700)))

    def assert_error(self, code, function, *args):
        with self.assertRaises(AppError) as context:
            function(*args)
        self.assertEqual(context.exception.code, code)

    def test_exact_half_open_vfr_interval(self):
        selected = select_frames(self.timeline, Fraction(1, 10), Fraction(7, 10))
        self.assertEqual([f.pts for f in selected], [5100, 5300])

    def test_non_frame_start_preserves_first_actual_frame(self):
        selected = select_frames(self.timeline, Fraction(100000000000000000001, 10**21), Fraction(7, 10))
        self.assertEqual([f.pts for f in selected], [5300])

    def test_observed_end_includes_last_duration(self):
        self.assertEqual(self.timeline.end, Fraction(4, 5))
        self.assertEqual(len(select_frames(self.timeline, Fraction(0), Fraction(4, 5))), 4)

    def test_subframe_empty_interval(self):
        self.assert_error("EMPTY_INTERVAL", select_frames, self.timeline, Fraction(31, 100), Fraction(69, 100))

    def test_end_not_silently_clipped(self):
        self.assert_error("INVALID_RANGE", select_frames, self.timeline, Fraction(0), Fraction(801, 1000))


class ContainerTests(unittest.TestCase):
    def inspect(self, data: bytes):
        with tempfile.TemporaryDirectory(prefix="crd-container-") as temporary:
            source = Path(temporary) / "input.mp4"
            source.write_bytes(data)
            validate_container(source)

    def test_self_contained_mp4(self):
        self.inspect(container())

    def test_external_references_are_rejected_before_probe(self):
        for target in (b"file:///private.mp4\0", b"https://example.invalid/media\0", b"\\\\server\\video.mp4\0"):
            with self.subTest(target=target), self.assertRaises(AppError) as context:
                self.inspect(container(b"\x00" * 4 + target))
            self.assertEqual(context.exception.code, "UNSUPPORTED_MEDIA")

    def test_self_flag_does_not_hide_reference_payload(self):
        with self.assertRaises(AppError):
            self.inspect(container(b"\x00\x00\x00\x01hidden-data"))

    def test_alias_reference_rejected(self):
        with self.assertRaises(AppError):
            self.inspect(container(entry_type=b"alis"))

    def test_playlist_named_mp4_is_rejected(self):
        with self.assertRaises(AppError):
            self.inspect(b"#EXTM3U\nhttps://example.invalid/segment.mp4\n")

    def test_truncated_and_overlong_boxes_rejected(self):
        for value in (container()[:-1], b"\x00\x00\x00\x07ftyp", struct.pack(">I4sQ", 1, b"ftyp", 2**63)):
            with self.subTest(value=value[:20]), self.assertRaises(AppError):
                self.inspect(value)

    def test_duplicate_moov_rejected(self):
        with self.assertRaises(AppError):
            self.inspect(container() + box(b"moov", b""))


class ProfileTests(unittest.TestCase):
    def video(self):
        return {"index": 0, "codec_type": "video", "codec_name": "h264", "pix_fmt": "yuv420p", "width": 320, "height": 180}

    def test_plain_8bit_h264(self):
        video, audio = _profile({"streams": [self.video()]})
        self.assertIsNone(audio)
        self.assertEqual(video["width"], 320)

    def test_hdr_and_10bit_rejected(self):
        for changes in ({"pix_fmt": "yuv420p10le"}, {"color_transfer": "smpte2084"}, {"color_transfer": "arib-std-b67"}):
            with self.subTest(changes=changes), self.assertRaises(AppError):
                _profile({"streams": [{**self.video(), **changes}]})

    def test_multiple_video_and_unknown_stream_rejected(self):
        for streams in ([self.video(), self.video()], [self.video(), {"codec_type": "data"}]):
            with self.assertRaises(AppError):
                _profile({"streams": streams})

    def test_rotation_cardinal_only(self):
        self.assertEqual(_rotation({"side_data_list": [{"rotation": -90}]}), 270)
        with self.assertRaises(AppError):
            _rotation({"side_data_list": [{"rotation": 45}]})


class DecodedTimelineTests(unittest.TestCase):
    def setUp(self):
        self.video = {"time_base": "1/1000", "width": 320, "height": 180, "pix_fmt": "yuv420p", "start_pts": 0, "duration_ts": 1000}
        self.tools = Toolchain(Path("ffmpeg"), Path("ffprobe"))

    def frames(self, pts):
        return [{"pts": p, "duration": 100, "width": 320, "height": 180, "pix_fmt": "yuv420p"} for p in pts]

    def decode(self, frames):
        with patch("cheating_racer_detect.media._json_tool", return_value={"frames": frames}):
            return _timeline(Path("input.mp4"), self.video, self.tools, "INVALID_TIMELINE")

    def test_missing_actual_pts_is_not_replaced_with_fps(self):
        frames = self.frames([0])
        del frames[0]["pts"]
        with self.assertRaises(AppError) as context:
            self.decode(frames)
        self.assertEqual(context.exception.code, "INVALID_TIMELINE")

    def test_duplicate_and_decreasing_pts_are_rejected(self):
        for pts in ([0, 100, 100], [0, 200, 100]):
            with self.subTest(pts=pts), self.assertRaises(AppError):
                self.decode(self.frames(pts))

    def test_midstream_format_change_is_rejected(self):
        frames = self.frames([0, 100])
        frames[1]["width"] = 640
        with self.assertRaises(AppError) as context:
            self.decode(frames)
        self.assertEqual(context.exception.code, "UNSUPPORTED_MEDIA")

    def test_last_frame_duration_can_come_from_stream_end(self):
        frames = self.frames([0, 400, 900])
        frames[-1]["duration"] = 0
        self.assertEqual(self.decode(frames).end, Fraction(1))


if __name__ == "__main__":
    unittest.main()
