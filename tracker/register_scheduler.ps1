# Register-ScheduledTask for DrugEx Thesis Time Tracker
$action = New-ScheduledTaskAction -Execute "python.exe" -Argument "`"C:\Users\kolar\Desktop\windows_native_projects\developer_expirience\thesis_tracker.py`" --sync-all"
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Friday -At 18:00
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName "DrugExThesisTimeTracker" -Action $action -Trigger $trigger -Settings $settings -Force
Write-Host "Successfully registered DrugExThesisTimeTracker task with StartWhenAvailable=True"
