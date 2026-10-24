; Inno Setup script for the Windows installer. packaging/build.py runs it after PyInstaller:
;   iscc /DAppVersion=0.1.0 /DBundleDir=dist\Vantage /Odist packaging\windows\vantage.iss
; Installs per user by default (no administrator rights needed, into %LOCALAPPDATA%\Programs\Vantage); the setup
; offers an all-users install too. Relative paths here are relative to this file's folder.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef BundleDir
  #define BundleDir "..\..\dist\Vantage"
#endif

[Setup]
AppId={{80403EE5-CC35-43C1-8A76-FA0ECDA337CD}
AppName=Vantage
AppVersion={#AppVersion}
AppVerName=Vantage {#AppVersion}
AppComments=Sensor rig planner for autonomous-driving data collection
AppPublisher=Vantage
AppPublisherURL=https://github.com/Sherin-SEF-AI/Vantage
AppSupportURL=https://github.com/Sherin-SEF-AI/Vantage/issues
AppUpdatesURL=https://github.com/Sherin-SEF-AI/Vantage/releases
VersionInfoVersion={#AppVersion}
VersionInfoProductName=Vantage
DefaultDirName={autopf}\Vantage
DefaultGroupName=Vantage
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputBaseFilename=Vantage-{#AppVersion}-windows-x64-setup
SetupIconFile=..\icons\vantage.ico
LicenseFile=..\..\LICENSE
UninstallDisplayIcon={app}\Vantage.exe
UninstallDisplayName=Vantage {#AppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ChangesAssociations=yes

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "assoc"; Description: "Open .vantage projects with Vantage"; GroupDescription: "File types:"

[Files]
Source: "{#BundleDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
; an update replaces the program folder's contents: files an older version had and this one does not go
Type: filesandordirs; Name: "{app}\_internal"

[Icons]
Name: "{group}\Vantage"; Filename: "{app}\Vantage.exe"
; cmd /k drops the first and the last quote of its command line unless exactly two quotes enclose a path with a space
; in it; the extra outer pair keeps the program's path quoted wherever it was installed
Name: "{group}\Vantage self-test"; Filename: "{cmd}"; Parameters: "/k """"{app}\vantage-cli.exe"" --self-test"""; IconFilename: "{app}\Vantage.exe"
Name: "{autodesktop}\Vantage"; Filename: "{app}\Vantage.exe"; Tasks: desktopicon

[Registry]
Root: HKA; Subkey: "Software\Classes\.vantage\OpenWithProgids"; ValueType: string; ValueName: "Vantage.Project"; ValueData: ""; Flags: uninsdeletevalue; Tasks: assoc
Root: HKA; Subkey: "Software\Classes\Vantage.Project"; ValueType: string; ValueName: ""; ValueData: "Vantage project"; Flags: uninsdeletekey; Tasks: assoc
Root: HKA; Subkey: "Software\Classes\Vantage.Project\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\Vantage.exe,0"; Tasks: assoc
Root: HKA; Subkey: "Software\Classes\Vantage.Project\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\Vantage.exe"" ""%1"""; Tasks: assoc

[Run]
Filename: "{app}\Vantage.exe"; Description: "{cm:LaunchProgram,Vantage}"; Flags: nowait postinstall skipifsilent
