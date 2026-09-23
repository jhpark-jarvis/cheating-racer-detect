param([switch]$DownloadTools)
$ErrorActionPreference = 'Stop'
$taskRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$taskVenv = Join-Path $taskRoot '.venv'
if (-not (Test-Path -LiteralPath (Join-Path $taskVenv 'Scripts/python.exe'))) {
    & py -3.13 -m venv --without-pip $taskVenv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.13 virtual environment creation failed.' }
}
if ($DownloadTools) {
    # Official FFmpeg download page links to this Windows build distributor.
    # The endpoint may move; a changed archive MUST fail this pinned checksum.
    $taskExpectedHash = '60f467265b1e312373dbcd92200c2618a74850f98d3d078e94296bb3fa2047ba'
    $taskToolsRoot = Join-Path $taskRoot '.tools'
    $taskTarget = Join-Path $taskToolsRoot 'ffmpeg-9.0.2-essentials_build'
    if (Test-Path -LiteralPath $taskTarget) {
        Write-Output 'FFmpeg directory already exists; bootstrap does not overwrite it.'
    } else {
        [void](New-Item -ItemType Directory -Path $taskToolsRoot -Force)
        $taskArchive = Join-Path $taskToolsRoot 'ffmpeg-9.0.2-essentials_build.zip'
        if (-not (Test-Path -LiteralPath $taskArchive)) {
            Invoke-WebRequest -Uri 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' -OutFile $taskArchive
        }
        $taskActualHash = (Get-FileHash -LiteralPath $taskArchive -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($taskActualHash -ne $taskExpectedHash) {
            throw 'Archive checksum mismatch. Do not execute it; review the pinned release before retrying.'
        }
        # Verify extraction targets remain inside .tools, then extract without overwriting.
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $taskZip = [System.IO.Compression.ZipFile]::OpenRead($taskArchive)
        try {
            $taskToolsPrefix = $taskToolsRoot.TrimEnd('\') + '\'
            foreach ($taskEntry in $taskZip.Entries) {
                $taskEntryPath = [System.IO.Path]::GetFullPath((Join-Path $taskToolsRoot $taskEntry.FullName))
                if (-not $taskEntryPath.StartsWith($taskToolsPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
                    throw 'Unsafe archive path.'
                }
            }
        } finally { $taskZip.Dispose() }
        [System.IO.Compression.ZipFile]::ExtractToDirectory($taskArchive, $taskToolsRoot)
        Write-Output "Prepared pinned FFmpeg archive SHA256=$taskActualHash"
    }
}
& (Join-Path $taskVenv 'Scripts/python.exe') --version
Write-Output 'Project-local environment ready. Global PATH and Python packages were not changed.'
