import winreg
from shutil import which
import os

userprofile = os.path.expandvars("$userprofile")
path = os.path.join(userprofile, "Dropbox", "lab", "c0_software", "moke", "diff_image_improved", "diff.bat")

for REG_PATH, CLASS, value in [
    [r"Software\Classes\*\shell\moke", winreg.REG_SZ, "process MOKE movie"],
    [r"Software\Classes\*\shell\moke\command", winreg.REG_EXPAND_SZ, f"\"{path}\" \"%1\""],
]:
    try:
        winreg.CreateKey(winreg.HKEY_CURRENT_USER, REG_PATH)
        registry_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, REG_PATH, 0, winreg.KEY_WRITE)
        winreg.SetValueEx(registry_key, "", 0, CLASS, value)
        winreg.CloseKey(registry_key)
    except WindowsError:
        pass