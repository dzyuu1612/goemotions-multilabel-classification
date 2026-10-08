param([Parameter(Mandatory=$true)][int]$GoEmotionsProcessId)
# Chỉ giữ hệ thống thức khi tiến trình thí nghiệm này còn chạy.
# Không đổi power plan, không ngăn tắt màn hình hoặc lệnh Sleep của người dùng.
Add-Type -TypeDefinition @'
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
public static class GoEmotionsAwake {
    [DllImport("kernel32.dll")]
    static extern uint SetThreadExecutionState(uint flags);
    public static void UntilExit(int id) {
        var target = Process.GetProcessById(id);
        if (SetThreadExecutionState(0x80000001) == 0)
            throw new InvalidOperationException("Cannot request temporary awake state");
        try { target.WaitForExit(); }
        finally { SetThreadExecutionState(0x80000000); }
    }
}
'@
Write-Output "Temporary awake request for experiment process $GoEmotionsProcessId"
[GoEmotionsAwake]::UntilExit($GoEmotionsProcessId)
Write-Output 'Experiment ended; temporary awake request released.'
