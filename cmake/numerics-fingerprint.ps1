# The numerics fingerprint for build.bat, which has no CMake: the same value cmake/build-info.cmake and tests/version.py compute (cmake/numerics-sources.txt).
# Each file in, in ordinal path order, as "path sha256\n" with CRLF read as LF, and the SHA-256 of those lines.
param([string]$Root)
$ins = @(); $outs = @()
foreach ($l in Get-Content (Join-Path $Root 'cmake/numerics-sources.txt')) {
    if ($l -match '^in +([^ |]+)') { $ins += $Matches[1] } elseif ($l -match '^out +([^ |]+)') { $outs += $Matches[1] }
}
$files = New-Object System.Collections.Generic.List[string]
foreach ($i in $ins) {
    $p = Join-Path $Root $i
    if (Test-Path $p -PathType Container) {
        foreach ($f in Get-ChildItem $p -Recurse -File -Force) { $files.Add($f.FullName.Substring($Root.Length).TrimStart('\', '/').Replace('\', '/')) }
    } else { $files.Add($i) }
}
$kept = New-Object System.Collections.Generic.List[string]
foreach ($f in $files) {
    $skip = $false
    foreach ($o in $outs) { if ($f -ceq $o -or $f.StartsWith($o + '/', [StringComparison]::Ordinal)) { $skip = $true } }
    if (-not $skip) { $kept.Add($f) }
}
$sorted = $kept.ToArray(); [Array]::Sort($sorted, [StringComparer]::Ordinal)
$latin = [Text.Encoding]::GetEncoding(28591)
$sha = [Security.Cryptography.SHA256]::Create()
$hex = { param($b) [BitConverter]::ToString($b).Replace('-', '').ToLowerInvariant() }
$text = New-Object Text.StringBuilder
foreach ($f in $sorted) {
    $folded = $latin.GetBytes($latin.GetString([IO.File]::ReadAllBytes((Join-Path $Root $f))).Replace("`r`n", "`n"))
    [void]$text.Append($f + ' ' + (& $hex $sha.ComputeHash($latin.GetBytes((& $hex $folded)))) + "`n")
}
Write-Output (& $hex $sha.ComputeHash($latin.GetBytes($text.ToString())))
