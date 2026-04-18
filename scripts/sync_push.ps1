param(
    [string]$RepoPath = "D:\AAAfinal\susu",
    [string]$Branch = "main",
    [string]$Message = ""
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git is not installed or not in PATH."
}

if (-not (Test-Path -LiteralPath $RepoPath)) {
    throw "RepoPath does not exist: $RepoPath"
}

Set-Location -LiteralPath $RepoPath

if (-not (Test-Path -LiteralPath (Join-Path $RepoPath ".git"))) {
    throw "Not a git repository: $RepoPath"
}

$statusOutput = git status --porcelain
if (-not $Message -or $Message.Trim().Length -eq 0) {
    $Message = "sync: update from local $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
}

if ($statusOutput) {
    git add -A
    git commit -m $Message
} else {
    Write-Host "[sync_push] No local changes to commit."
}

$originUrl = git remote get-url origin 2>$null
if (-not $originUrl) {
    throw "Remote 'origin' is not configured. Run: git remote add origin <repo-url>"
}

git push origin $Branch

Write-Host "[sync_push] Done."
Write-Host "  Repo  : $RepoPath"
Write-Host "  Branch: $Branch"
Write-Host "  Origin: $originUrl"
