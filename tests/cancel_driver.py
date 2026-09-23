"""Hidden-console F09 driver: real Windows console event, owned processes only.

The integration test starts this module with CREATE_NEW_CONSOLE and SW_HIDE.
This driver gives the CLI its own console process group. CTRL_BREAK_EVENT goes
only to that group; it never broadcasts to the user's console or services.
Observed media children are held by OS handles, avoiding PID reuse during the
termination checks / emergency cleanup. No mocks replace the product process.
With --force the driver terminates only those owned processes, verifies retained
crash residue across retry, then removes its identity-checked fixture residue.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time


class ProcessEntry(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG),
        ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260),
    ]


def windows_api():
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.CreateToolhelp32Snapshot.argtypes = (wintypes.DWORD, wintypes.DWORD)
    api.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    api.Process32FirstW.argtypes = (wintypes.HANDLE, ctypes.POINTER(ProcessEntry))
    api.Process32NextW.argtypes = (wintypes.HANDLE, ctypes.POINTER(ProcessEntry))
    api.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    api.OpenProcess.restype = wintypes.HANDLE
    api.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    api.WaitForSingleObject.restype = wintypes.DWORD
    api.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
    api.CloseHandle.argtypes = (wintypes.HANDLE,)
    return api


def children(api, parent_pid: int) -> list[tuple[int, str]]:
    snapshot = api.CreateToolhelp32Snapshot(2, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        raise OSError(ctypes.get_last_error(), "Process enumeration unavailable")
    entry = ProcessEntry()
    entry.dwSize = ctypes.sizeof(entry)
    entries = []
    try:
        available = api.Process32FirstW(snapshot, ctypes.byref(entry))
        while available:
            entries.append((entry.th32ProcessID, entry.th32ParentProcessID, entry.szExeFile))
            available = api.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        api.CloseHandle(snapshot)
    # Windows venv python.exe may be a launcher, so FFmpeg can be a grandchild
    # of Popen.pid. Establish ancestry from this one coherent OS snapshot.
    owned = {parent_pid}
    found = []
    while True:
        added = [(pid, name) for pid, parent, name in entries if parent in owned and pid not in owned]
        if not added:
            return found
        found.extend(added)
        owned.update(pid for pid, _ in added)


def main() -> int:
    source, output, tools_dir = map(Path, sys.argv[1:4])
    force = sys.argv[4:] == ['--force']
    api = windows_api()
    process = None
    retry_process = None
    observed = {}
    retry_observed = {}
    environment = os.environ.copy()
    environment["CRD_FFMPEG_DIR"] = str(tools_dir)
    command = [
        sys.executable, "-m", "cheating_racer_detect", "clip", "--input", str(source),
        "--start", "0.35", "--end", "2.35", "--output", str(output),
    ]
    before = set(output.parent.glob(".crd-*.partial"))

    def observe_children(parent, tracked):
        for pid, name in children(api, parent.pid):
            if pid in tracked:
                continue
            handle = api.OpenProcess(0x100000 | 0x0001, False, pid)  # SYNCHRONIZE | TERMINATE
            if handle:
                tracked[pid] = (handle, name)

    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        deadline = time.monotonic() + 15
        temporary_seen = False
        active_media = False
        while process.poll() is None and time.monotonic() < deadline:
            temporary_seen = bool(set(output.parent.glob(".crd-*.partial")) - before)
            observe_children(process, observed)
            active_media = any(
                name.lower() in {"ffmpeg.exe", "ffprobe.exe"} and api.WaitForSingleObject(handle, 0) == 258
                for handle, name in observed.values()
            )
            if temporary_seen and active_media:
                break
            time.sleep(0.005)
        if not temporary_seen or not active_media or process.poll() is not None:
            exit_code = process.poll()
            code = None
            if exit_code is not None:
                _, diagnostic = process.communicate()
                try:
                    code = json.loads(diagnostic.decode("utf-8")).get("code")
                except (ValueError, UnicodeError):
                    code = "non_json_error" if diagnostic else None
            raise RuntimeError(
                "Could not observe an active owned media process before completion: "
                f"exit={exit_code}, code={code}, temporary_seen={temporary_seen}, observed={len(observed)}"
            )
        leftovers = {}
        if force:
            for path in set(output.parent.glob('.crd-*.partial')) - before:
                stat = path.stat()
                leftovers[path] = (stat.st_dev, stat.st_ino)
            # Kill the launcher/interpreter before media so graceful exception
            # handling cannot clean the crash fixture. Only held owned handles.
            observe_children(process, observed)
            process.kill()
            for handle, name in sorted(observed.values(), key=lambda item: item[1].lower() in {'ffmpeg.exe', 'ffprobe.exe'}):
                if api.WaitForSingleObject(handle, 0) == 258:
                    if not api.TerminateProcess(handle, 1):
                        raise OSError(ctypes.get_last_error(), 'Owned process termination failed')
            stdout, stderr = process.communicate(timeout=15)
            response = {'code': None}
            if not process.returncode or output.exists() or not leftovers:
                raise AssertionError('Forced termination unexpectedly published or left no crash fixture')
            if set(output.parent.glob('.crd-*.partial')) != before | set(leftovers):
                raise AssertionError('Crash residue does not match captured owned directories')
        else:
            os.kill(process.pid, signal.CTRL_BREAK_EVENT)
            stdout, stderr = process.communicate(timeout=15)
            response = json.loads(stderr.decode("utf-8"))
            if process.returncode != 130 or response.get("code") != "CANCELLED":
                raise AssertionError(f"Expected cancellation; got exit={process.returncode}, code={response.get('code')}")
            if output.exists() or set(output.parent.glob(".crd-*.partial")) != before:
                raise AssertionError("Cancellation left a final result or owned temporary directory")
        for handle, _ in observed.values():
            if api.WaitForSingleObject(handle, 3000) != 0:
                raise AssertionError("An owned media child survived cancellation")
        retry_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                         env=environment, creationflags=subprocess.CREATE_NO_WINDOW)
        retry_deadline = time.monotonic() + 30
        while retry_process.poll() is None:
            observe_children(retry_process, retry_observed)
            if time.monotonic() >= retry_deadline:
                raise TimeoutError("Retry exceeded its 30-second test limit")
            time.sleep(0.005)
        retry_process.communicate()
        if retry_process.returncode or not (output / "result.json").is_file():
            raise AssertionError("Retry of the same output after cancellation failed")
        if any(api.WaitForSingleObject(handle, 3000) != 0 for handle, _ in retry_observed.values()):
            raise AssertionError("An owned retry child survived completion")
        if force:
            if set(output.parent.glob('.crd-*.partial')) != before | set(leftovers):
                raise AssertionError('Retry changed another invocation crash residue')
            for path, identity in leftovers.items():
                stat = path.lstat()
                if (path.resolve().parent != output.parent.resolve() or path.is_symlink()
                        or getattr(stat, 'st_file_attributes', 0) & 0x400
                        or (stat.st_dev, stat.st_ino) != identity):
                    raise AssertionError('Crash residue ownership changed; refuse cleanup')
                shutil.rmtree(path)
            if set(output.parent.glob('.crd-*.partial')) != before:
                raise AssertionError('Test-owned crash residue cleanup failed')
        print(json.dumps({
            "status": "passed", "event": "TerminateProcess" if force else "CTRL_BREAK_EVENT", "cancel_exit": process.returncode,
            "cancel_code": response["code"], "cli_pid": process.pid,
            "observed_children": [{"pid": pid, "name": value[1]} for pid, value in observed.items()],
            "children_exited": True, "temporary_cleaned": True, "retry_exit": retry_process.returncode,
            "original_console_untouched": True,
            "crash_residue_preserved_on_retry": bool(leftovers),
        }))
        return 0
    except Exception as error:
        print(json.dumps({"status": "failed", "kind": type(error).__name__, "reason": str(error)}))
        return 1
    finally:
        # Emergency cleanup is restricted to this driver's exact process object
        # and the child handles opened while their parent was that process.
        for owned_process in (process, retry_process):
            if owned_process is not None and owned_process.poll() is None:
                # Capture launcher descendants once more before killing their
                # parent, so emergency teardown cannot leave its interpreter.
                observe_children(owned_process, observed if owned_process is process else retry_observed)
                owned_process.kill()
        for handle, _ in list(observed.values()) + list(retry_observed.values()):
            if api.WaitForSingleObject(handle, 0) == 258:
                api.TerminateProcess(handle, 1)
                api.WaitForSingleObject(handle, 3000)
            api.CloseHandle(handle)
        for owned_process in (process, retry_process):
            if owned_process is not None:
                owned_process.communicate()


if __name__ == "__main__":
    raise SystemExit(main())
