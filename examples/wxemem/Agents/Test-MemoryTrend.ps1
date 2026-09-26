[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$generator = Join-Path $PSScriptRoot 'Generate-MemoryTrend.ps1'
$testRoot = Join-Path ([IO.Path]::GetTempPath()) ('memory-trend-test-' + [guid]::NewGuid())
$daily = Join-Path $testRoot 'daily'
$output = Join-Path $testRoot 'output'
$list = Join-Path $testRoot 'builds.txt'

function Assert-True([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}

function Write-Capture([string]$Build, [string]$Stage, [long]$Used, [string]$Name = 'wxemem_json.txt') {
    $dir = Join-Path (Join-Path (Join-Path $daily $Build) 'Kennan\WXE') $Stage
    $null = New-Item -ItemType Directory -Path $dir -Force
    @{physical = @{used_bytes = $Used; total_bytes = 16GB}} |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $dir $Name) -Encoding UTF8
}

function Expect-Error([scriptblock]$Action, [string]$Pattern) {
    try { & $Action } catch {
        if ($_.Exception.Message -notlike $Pattern) { throw }
        return
    }
    throw "Expected an error matching '$Pattern'."
}

try {
    $b1 = '100.1.260101-1700'
    $b2 = '100.2.260102-1700'
    $b3 = '100.3.260103-1700'
    $b4 = '100.4.260104-1700'
    $b5 = '100.5.260105-1700'
    Write-Capture $b1 'prepnp' 1GB
    Write-Capture $b1 'postpnpxboxpcapp' 3GB
    Write-Capture $b2 'prepnpxboxapp' 2GB
    Write-Capture $b2 'postpnpxboxapp\Houseman' 8GB
    Write-Capture $b3 'afterpnp' 4GB
    Write-Capture $b3 'prepnp\Omni' 9GB
    Write-Capture $b4 'nopnp' 0
    Write-Capture $b4 'pnp' 5GB
    Write-Capture $b1 'signin' 6GB
    Write-Capture $b2 'xboxappsignin' 7GB
    Write-Capture $b3 'signin\Omni' 10GB
    Write-Capture $b3 'postsignin' 11GB
    Write-Capture $b4 'presignin' 12GB
    Write-Capture $b4 'steamsignin' 13GB
    Write-Capture $b5 'signin' 8GB
    $null = & $generator -DailyRoot $daily -OutputDirectory $output -DataOnly -WarningAction SilentlyContinue
    $rows = @(Import-Csv (Join-Path $output 'memory-trend.csv'))
    Assert-True ($rows.Count -eq 5) 'Expected five build categories.'
    Assert-True ($rows[0].build -eq $b1 -and [long]$rows[0].postpnp_used_bytes -eq 3GB) 'postpnpxboxpcapp alias failed.'
    Assert-True ($rows[1].status -eq 'included' -and $rows[1].postpnp_used_bytes -eq '') 'Missing post-PnP must remain blank.'
    Assert-True ($rows[2].used_bytes -eq '' -and [long]$rows[2].postpnp_used_bytes -eq 4GB) 'Post-only sample was lost or nested pre sample included.'
    Assert-True ($rows[3].used_bytes -eq '0' -and $rows[3].status -eq 'included') 'Real zero must not become missing.'
    Assert-True ([long]$rows[0].signin_used_bytes -eq 6GB -and [long]$rows[1].signin_used_bytes -eq 7GB) 'Sign-in aliases failed.'
    Assert-True ([long]$rows[2].signin_used_bytes -eq 11GB) 'postsignin alias failed or nested capture was included.'
    Assert-True ($rows[3].signin_used_bytes -eq '') 'Other sign-in stages must not be included.'
    Assert-True ($rows[4].used_bytes -eq '' -and $rows[4].postpnp_used_bytes -eq '' -and [long]$rows[4].signin_used_bytes -eq 8GB) 'Sign-in-only sample lost or mixed with other stages.'
    $coverage = Get-Content (Join-Path $output 'memory-trend.coverage.json') -Raw | ConvertFrom-Json
    Assert-True ($coverage.included_builds -eq 3 -and $coverage.postpnp_included_builds -eq 3) 'Incorrect stage coverage.'
    Assert-True ($coverage.signin_included_builds -eq 4 -and $coverage.signin_missing_builds.Count -eq 1) 'Incorrect sign-in coverage.'

    Remove-Item -LiteralPath (Join-Path $daily "$b3\Kennan\WXE\postsignin\wxemem_json.txt")
    @($b3, $b1) | Set-Content -LiteralPath $list
    $null = & $generator -DailyRoot $daily -OutputDirectory $output -BuildList $list -DataOnly -WarningAction SilentlyContinue
    $rows = @(Import-Csv (Join-Path $output 'memory-trend.csv'))
    Assert-True ($rows.Count -eq 2 -and $rows[0].build -eq $b1 -and $rows[1].build -eq $b3) 'BuildList chronology/selection failed.'
    $b3 | Set-Content -LiteralPath $list
    $null = & $generator -DailyRoot $daily -OutputDirectory $output -BuildList $list -DataOnly -WarningAction SilentlyContinue
    $rows = @(Import-Csv (Join-Path $output 'memory-trend.csv'))
    Assert-True ($rows.Count -eq 1 -and $rows[0].postpnp_status -eq 'included') 'Post-only selection must succeed.'
    Assert-True ($rows[0].signin_used_bytes -eq '') 'Nested sign-in capture must not fill a missing direct sample.'

    $b5 | Set-Content -LiteralPath $list
    $null = & $generator -DailyRoot $daily -OutputDirectory $output -BuildList $list -DataOnly -WarningAction SilentlyContinue
    $rows = @(Import-Csv (Join-Path $output 'memory-trend.csv'))
    Assert-True ($rows.Count -eq 1 -and $rows[0].signin_status -eq 'included') 'Sign-in-only selection must succeed.'
    Write-Capture $b1 'xboxappsignin' 9GB
    Expect-Error { & $generator -DailyRoot $daily -OutputDirectory $output -DataOnly } '*Ambiguous sign-in*'
    Remove-Item -LiteralPath (Join-Path $daily "$b1\Kennan\WXE\xboxappsignin\wxemem_json.txt")
    Write-Capture $b1 'postsignin' 9GB
    Expect-Error { & $generator -DailyRoot $daily -OutputDirectory $output -DataOnly } '*Ambiguous sign-in*'
    Remove-Item -LiteralPath (Join-Path $daily "$b1\Kennan\WXE\postsignin\wxemem_json.txt")
    '{invalid' | Set-Content -LiteralPath (Join-Path $daily "$b1\Kennan\WXE\signin\wxemem_json.txt")
    Expect-Error { & $generator -DailyRoot $daily -OutputDirectory $output -DataOnly } '*Cannot parse capture*'
    Write-Capture $b1 'signin' 6GB
    Write-Capture $b1 'postpnpxboxpcapp' 6GB 'wxemem_other_json.txt'
    Expect-Error { & $generator -DailyRoot $daily -OutputDirectory $output -DataOnly } '*Ambiguous post-PnP*'
    Remove-Item -LiteralPath (Join-Path $daily "$b1\Kennan\WXE\postpnpxboxpcapp\wxemem_other_json.txt")
    '{invalid' | Set-Content -LiteralPath (Join-Path $daily "$b1\Kennan\WXE\postpnpxboxpcapp\wxemem_json.txt")
    Expect-Error { & $generator -DailyRoot $daily -OutputDirectory $output -DataOnly } '*Cannot parse capture*'
    Write-Output 'Passed: all three stages, independent gaps, excluded aliases/nested data, real zero, build selection, post-only/sign-in-only data, ambiguity, invalid JSON.'
} finally {
    if (Test-Path -LiteralPath $testRoot) {
        Get-ChildItem -LiteralPath $testRoot -Recurse -File | ForEach-Object { Remove-Item -LiteralPath $_.FullName }
        Get-ChildItem -LiteralPath $testRoot -Recurse -Directory |
            Sort-Object { $_.FullName.Length } -Descending |
            ForEach-Object { Remove-Item -LiteralPath $_.FullName }
        Remove-Item -LiteralPath $testRoot
    }
}
