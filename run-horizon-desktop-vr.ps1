param(
    [string]$GameDir,
    [int]$GuestMemoryMB = 8192,
    [int]$Width = 1280,
    [int]$Height = 720
)

$ErrorActionPreference = "Stop"
$exe = Join-Path $PSScriptRoot "_Build\windows-no-qt3\kyty_emulator.exe"
if (-not (Test-Path -LiteralPath $exe)) {
    throw "KytyPS5 has not been built: $exe"
}
if ([string]::IsNullOrWhiteSpace($GameDir)) {
    if ([Threading.Thread]::CurrentThread.GetApartmentState() -ne [Threading.ApartmentState]::STA) {
        throw "Folder selection requires STA PowerShell. Start with powershell.exe -STA or pass -GameDir."
    }
    Add-Type -AssemblyName System.Windows.Forms
    $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
    $dialog.Description = "Select the Horizon Call of the Mountain game folder containing eboot.bin"
    $dialog.ShowNewFolderButton = $false
    try {
        if ($dialog.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) {
            throw "Game folder selection cancelled. Pass -GameDir to skip the picker."
        }
        $GameDir = $dialog.SelectedPath
    } finally {
        $dialog.Dispose()
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $GameDir "eboot.bin"))) {
    throw "Horizon Call of the Mountain eboot.bin was not found in: $GameDir"
}
if ($GuestMemoryMB -lt 2048 -or $GuestMemoryMB -gt 13824) {
    throw "GuestMemoryMB must be between 2048 and 13824"
}

$env:KYTY_GUEST_BACKING_MB = [string]$GuestMemoryMB

Write-Host "Horizon Call of the Mountain - experimental virtual PS VR2 desktop mode"
Write-Host "Gameplay, including climbing and bow use, is not verified; loading/rendering work continues."
Write-Host "Default keyboard/mouse controls:"
Write-Host "F6: toggle cropped left-eye/full-panel desktop mirror"
Write-Host "F7: mouse controls right stick"
Write-Host "F8: capture/release mouse for headset look"
Write-Host "F9: recenter headset and reset both hand positions, reach and wrist aim"
Write-Host "F2/F3: toggle left/right hand position control (mouse XY; wheel depth)"
Write-Host "F5/F10: toggle left/right wrist aim with the mouse (Shift+F10 also selects left)"
Write-Host "Positioned hands stay in place when you turn your head or release mouse control; F9 resets them."
Write-Host "Wheel: selected hand depth/reach; right hand if none selected. Shift+wheel: left (Shift is also L3)."
Write-Host "WASD: left stick; TFGH: right stick; JIKL: face buttons; Enter: options"
Write-Host "Horizon primary grip/trigger: Z or middle click = left L2; C or left click = right R2"
Write-Host "Q/E (L1/R1), or right click (R1): alternate grip/standing jump, depending on game context"
Write-Host "Custom bindings replace game-input defaults; bindings assigned to F2/F3/F5 take priority."
Write-Host "Physical gamepads are also supported through the normal KytyPS5 mappings."

& $exe --game $GameDir --vr --skip-notice-screen true --screen-width $Width --screen-height $Height
exit $LASTEXITCODE
