<#
Deploy kode (yang ter-commit, HEAD) ke Jetson dalam satu perintah. Jetson tidak memakai git: ~/gmf-forklift
adalah salinan. IP berubah per jaringan (gsih 192.168.88.x, hotspot 172.28.206.x, USB-C 192.168.55.1).

  powershell -File tools/jetson/deploy.ps1 -JetsonHost 192.168.88.102
  powershell -File tools/jetson/deploy.ps1 -JetsonHost 192.168.88.102 -Restart   # + restart detektor

Langkah: cadangan src/configs/tools di Jetson -> kirim arsip git HEAD -> model YuNet jika belum ada ->
cek di Jetson (OpenCV FaceDetectorYN, import + config overlay) -> opsional restart lewat tools/jetson/live.sh.
Bobot/engine YOLO TIDAK dikirim (engine dibangun di Jetson).
#>
param(
    [Parameter(Mandatory = $true)][string]$JetsonHost,
    [string]$User = "jetson",
    [string]$Key = "$env:USERPROFILE\.ssh\gmf_jetson",
    [string]$RemoteDir = "/home/jetson/gmf-forklift",
    [switch]$Restart
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
$target = "$User@$JetsonHost"
$sshOpts = @("-i", $Key, "-o", "ConnectTimeout=8", "-o", "BatchMode=yes")
function Invoke-Remote([string]$cmd) {
    & ssh @sshOpts $target $cmd
    if ($LASTEXITCODE -ne 0) { throw "ssh gagal ($LASTEXITCODE): $cmd" }
}

$dirty = git -C $repo status --porcelain -- src configs tools eval
if ($dirty) { Write-Warning "Ada perubahan belum di-commit (tidak ikut terkirim):`n$dirty" }
$rev = git -C $repo rev-parse --short HEAD
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$tar = Join-Path $env:TEMP "gmf-deploy-$rev.tar"

Write-Host "[1/5] arsip HEAD $rev"
git -C $repo archive --format=tar -o $tar HEAD src configs tools eval/replay_eval.py
if ($LASTEXITCODE -ne 0) { throw "git archive gagal" }

Write-Host "[2/5] cadangan di Jetson -> backups/pre-$stamp.tgz"
Invoke-Remote "cd $RemoteDir && mkdir -p backups && tar -czf backups/pre-$stamp.tgz src configs tools"

Write-Host "[3/5] kirim + ekstrak"
& scp @sshOpts $tar "${target}:/tmp/gmf-deploy.tar"
if ($LASTEXITCODE -ne 0) { throw "scp gagal" }
Invoke-Remote "cd $RemoteDir && tar -xf /tmp/gmf-deploy.tar && rm /tmp/gmf-deploy.tar && echo $rev > DEPLOYED_REV"
Remove-Item $tar

Write-Host "[4/5] model YuNet"
$yunet = "models/face/face_detection_yunet_2023mar.onnx"
& ssh @sshOpts $target "test -f $RemoteDir/$yunet"
if ($LASTEXITCODE -ne 0) {
    Invoke-Remote "mkdir -p $RemoteDir/models/face"
    & scp @sshOpts (Join-Path $repo $yunet) "${target}:$RemoteDir/$yunet"
    if ($LASTEXITCODE -ne 0) { throw "scp YuNet gagal" }
}

Write-Host "[5/5] cek di Jetson"
$check = @"
cd $RemoteDir && . .venv/bin/activate 2>/dev/null; python - <<'EOF'
import cv2
from src.config import Config, load_yaml
assert hasattr(cv2, 'FaceDetectorYN'), 'OpenCV ' + cv2.__version__ + ' tanpa FaceDetectorYN (butuh >= 4.8)'
cfg = Config.load(overrides=load_yaml('configs/jetson.yaml'))
fc = cfg.runtime['cabin']['face']
from src.cabin.face import make_face_model
m = make_face_model(fc)
print('OK opencv', cv2.__version__, '| face', type(m).__name__, '| crop', fc.get('crop', {}).get('enabled'),
      '| yolo async', cfg.runtime['cabin']['yolo'].get('async'),
      '| rekam', cfg.runtime.get('continuous_recording', {}).get('enabled'))
EOF
"@
Invoke-Remote ($check -replace "`r", "")

if ($Restart) {
    Write-Host "restart detektor"
    Invoke-Remote "cd $RemoteDir && bash tools/jetson/live.sh stop; bash tools/jetson/live.sh start"
}
Write-Host "Selesai: $rev di $JetsonHost (cadangan backups/pre-$stamp.tgz)"
