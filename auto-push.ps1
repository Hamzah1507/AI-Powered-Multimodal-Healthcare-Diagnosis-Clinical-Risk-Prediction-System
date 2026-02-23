# Prevent multiple parallel runs
$lockFile = ".autopush.lock"

if (Test-Path $lockFile) {
    Write-Host "Auto-push already running, skipping..."
    exit
}

New-Item $lockFile -ItemType File | Out-Null

try {
    # Debounce delay (fast but stable)
    Start-Sleep -Milliseconds 1500

    git add -A

    $changesRaw = git diff --name-only --cached

    if (-not $changesRaw) {
        Write-Host "No staged changes."
        return
    }

    $files = $changesRaw -split "`n"
    $files = $files | ForEach-Object { $_.Trim() -replace '\\', '/' }

    $changedFile = $files | Select-Object -First 1

    if (-not $changedFile) {
        Write-Host "No file change detected."
        return
    }

    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($changedFile)
    $readableName = $baseName -replace '_', ' '

    $commitMessage = "feat: $readableName updated"

    Write-Host "Auto commit message: $commitMessage"

    git commit -m "$commitMessage"
    git push
}
finally {
    Remove-Item $lockFile -ErrorAction SilentlyContinue
}