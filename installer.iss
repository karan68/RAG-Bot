; Inno Setup Script for Device Chatbot
; This creates a professional Windows installer (.exe)
;
; Requirements:
;   1. Build the app first: python build_installer.py
;   2. Install Inno Setup: https://jrsoftware.org/isinfo.php
;   3. Compile this script with Inno Setup Compiler

#define MyAppName "Device Chatbot"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Device Chatbot"
#define MyAppExeName "DeviceChatbot.exe"
#define MyAppURL "https://github.com/device-chatbot"

[Setup]
; Basic info
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}

; Install location
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes

; Output settings
OutputDir=installer_output
OutputBaseFilename=DeviceChatbot_Setup_v{#MyAppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes

; Appearance
WizardStyle=modern
; UninstallDisplayIcon={app}\{#MyAppExeName}

; Permissions
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

; Other
SetupIconFile=
DisableWelcomePage=no
AllowNoIcons=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Main application files from PyInstaller dist folder
Source: "dist\DeviceChatbot\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Messages]
WelcomeLabel1=Welcome to {#MyAppName} Setup
WelcomeLabel2=This will install {#MyAppName} v{#MyAppVersion} on your computer.%n%n{#MyAppName} is a local AI chatbot for device specifications. It runs entirely offline using local AI models.%n%nIMPORTANT: You need Ollama installed to run this app. Download from https://ollama.ai

[Code]
// Check if Ollama is installed
function IsOllamaInstalled(): Boolean;
var
  ResultCode: Integer;
begin
  Result := Exec('cmd.exe', '/c ollama --version', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0);
end;

// Show warning if Ollama not installed
procedure CurPageChanged(CurPageID: Integer);
begin
  if CurPageID = wpReady then
  begin
    if not IsOllamaInstalled() then
    begin
      MsgBox('Ollama is not detected on your system.'#13#10#13#10 +
             'Device Chatbot requires Ollama to run AI models locally.'#13#10#13#10 +
             'Please install Ollama from https://ollama.ai before or after this installation.'#13#10#13#10 +
             'The app will prompt you to download models (~1.3 GB) on first run.',
             mbInformation, MB_OK);
    end;
  end;
end;

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
