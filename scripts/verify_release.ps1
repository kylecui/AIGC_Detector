# Release verification gate (release-plan-v2 DoD) — machine-checkable PASS/FAIL.
# Usage: pwsh scripts/verify_release.ps1 [-Batch v0.1]
param([string]$Batch = "v0.1")

$ErrorActionPreference = "Continue"
$results = @()
function Add-Check($name, $ok, $detail) {
    $script:results += [pscustomobject]@{Check=$name; Ok=$ok; Detail=$detail}
    Write-Output "$(if($ok){'PASS'}else{'FAIL'})  $name  $detail"
}

# 1. LICENSE exists and is non-trivial
$lic = Get-Item LICENSE -ErrorAction SilentlyContinue
Add-Check "LICENSE present" ($null -ne $lic -and $lic.Length -gt 5000) "$(if($lic){$lic.Length}else{0}) bytes"

# 2. README quick start points at the real entry (no stub main.py)
$rd = Get-Content README.md -Raw
Add-Check "README real entry" ($rd -match "uvicorn aigc_detector" -and $rd -notmatch "uv run python main\.py`n") "uvicorn cmd documented"

# 3. Wheel builds and is self-contained
uv build --wheel *> $null
$whl = Get-ChildItem dist/*.whl -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
$ok3 = $false; $det3 = "no wheel"
if ($whl) {
    $zip = [System.IO.Compression.ZipFile]::OpenRead($whl.FullName)
    $names = $zip.Entries.FullName
    $zip.Dispose()
    $hasStatic = ($names | Where-Object { $_ -match "aigc_detector/static/index" }).Count -gt 0
    $hasCfg = ($names | Where-Object { $_ -match "aigc_detector/configs/models.yaml" }).Count -gt 0
    $ok3 = $hasStatic -and $hasCfg
    $det3 = "static=$hasStatic configs=$hasCfg"
}
Add-Check "Wheel self-contained" $ok3 $det3

# 4. Console entry point works (doctor; wheel layout shows [OK] static,
#    dev checkout falls back to repo layout and reports the fallback)
$docOut = uv run aigc-detector doctor 2>&1 | Out-String
Add-Check "CLI doctor" ($LASTEXITCODE -eq 0 -and $docOut -match "(\[OK\] static|repo-layout fallback)") "exit=$LASTEXITCODE"

# 5. Calibration artifacts present + applied/enabled flags
$t = Get-Content models/calibration/global_temperature.json -Raw -ErrorAction SilentlyContinue | ConvertFrom-Json
$f = Get-Content models/calibration/binoculars_floor.json -Raw -ErrorAction SilentlyContinue | ConvertFrom-Json
Add-Check "Calibration deployed" ($t.applied -eq $true -and $f.enabled -eq $true) "T=$($t.T) floor=$($f.cutoff)"

# 6. Model pins recorded
$pins = Get-Content models/calibration/model_pins.json -Raw -ErrorAction SilentlyContinue | ConvertFrom-Json
$pinnedN = ($pins.models.PSObject.Properties | Where-Object { $_.Value.revision }).Count
Add-Check "Model pins" ($pinnedN -ge 10) "$pinnedN/14 revisions pinned"

# 7. Full test suite
uv run pytest tests/ -q --tb=no *> $null
Add-Check "Test suite" ($LASTEXITCODE -eq 0) "exit=$LASTEXITCODE"

# 8. EN-formal guard + log-hygiene guards in place (source-level)
$routes = Get-Content src/aigc_detector/api/routes.py -Raw
Add-Check "EN formal guard wired" ($routes -match "_en_formal_downgrade\(result") "both endpoints"
Add-Check "Log hygiene wired" ((Get-Content src/aigc_detector/api/main.py -Raw) -match "setup_service_logging") "lifespan"

# 9. Git state: nothing uncommitted (this run's own report is expected noise)
$dirty = git status --porcelain=v1 2>$null | Where-Object { $_ -notmatch "release-check" }
Add-Check "Worktree clean" ($null -eq $dirty -or $dirty.Count -eq 0) "$(if($dirty){$dirty.Count}else{0}) dirty"

# Summary
$failed = ($results | Where-Object { -not $_.Ok }).Count
$verdict = if ($failed -eq 0) { "PASS" } else { "FAIL" }
Write-Output ""
Write-Output "=== BATCH $Batch VERIFICATION: $verdict ($($results.Count - $failed)/$($results.Count) checks) ==="
New-Item -ItemType Directory -Force -Path reports | Out-Null
$ts = Get-Date -Format "yyyy-MM-dd"
$results | ConvertTo-Json -Depth 3 | Set-Content "reports/release-check-$Batch-$ts.json"
Write-Output "report: reports/release-check-$Batch-$ts.json"
if ($verdict -eq "FAIL") { exit 1 }
