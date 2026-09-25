$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$downstreamPid = 70980
$log = Join-Path $root 'resume_three_domain_after_milling.log'
"Watcher started $(Get-Date -Format o); waiting for Milling downstream PID $downstreamPid" | Set-Content -LiteralPath $log
while (Get-Process -Id $downstreamPid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 20
}
"Milling downstream finished $(Get-Date -Format o); starting three-domain resume" | Add-Content -LiteralPath $log
$python = 'python'
$pkg = Join-Path $root 'three_domain'
$train = Join-Path $pkg 'code\train_joint.py'
Start-Process -FilePath $python -ArgumentList @('-u', $train, '--resume') -WorkingDirectory $pkg -RedirectStandardOutput (Join-Path $pkg 'stdout.log') -RedirectStandardError (Join-Path $pkg 'stderr.log') -WindowStyle Hidden
"Three-domain resume launched $(Get-Date -Format o)" | Add-Content -LiteralPath $log
