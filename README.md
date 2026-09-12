# MAES

基于 MaaFramework 5.12.2 与 ProjectInterface V2 ，依托Deepseek和Codex实现的《偶像梦幻祭2》打歌自动化项目。
当前版本 **v1.0.0-Stable**，聚焦「临时打歌」与「打歌点位校准」。

## 功能范围

- **打歌点位校准**：在演唱会进行页识别 7/9 个判定点并保存本地校准。
- **临时打歌**：7 轨实时跟踪与自动打歌（Tap / 长按 / 四向划动 / 变轨长按 / 尾端划动）。
- **临时打歌（9轨实验）**：仅做校准与候选识别预检，不发送正式打歌动作。
- **打歌触控能力检测**：触控异常时的诊断工具（歌曲暂停页运行），正常打歌无需预先运行。

> 启动游戏、日常、事务所、星光演唱会、活动与送抽等任务暂未提供，后续版本逐步加入。

## 使用（普通用户）

1. 从本仓库 Releases 下载最新版本压缩包并解压到任意目录。
2. 系统要求：Windows 10/11 x64；.NET Desktop Runtime 10；VC++ 2015-2022 Redistributable；安卓模拟器（支持 ADB）或实机设备。
   缺少运行库时，以管理员身份运行包内 `DependencySetup_依赖库安装_win.bat` 可自动安装。
3. 打开 `MFAAvalonia.exe`，添加你的模拟器设备；确认设备“输入模式”为 Default（打歌必需）。
4. 启动游戏并进入演唱会进行页（画面需为 1280×720 或者相同比例）。
5. 第一次运行，先勾选「打歌点位校准」和「临时打歌」，「打歌点位校准」在识别出 7 个判定点后完成，理论用时较短。在校准完成后，「临时打歌」功能将开始运行，整首歌将自动进行；结束时以 LIVE 结算检测为准
6. 之后的运行，理论上只需要运行「临时打歌」，整首歌将自动进行；结束时以 LIVE 结算检测为准。
7. 若打歌出现触控异常：暂停歌曲后运行「打歌触控能力检测」定位问题。

**日志与数据**：任务日志位于 `logs/`（含 `tap-traces`）；错误截图位于 `debug/on_error`；用户数据（校准、触控状态、最近结果）保存在本机 `%LOCALAPPDATA%\MAES`，可用环境变量 `MAES_DATA_DIR` 指定其他目录。

## 已知限制

- 仅支持 1280×720 的演唱会进行页；其他分辨率需要自行适配。
- 打歌从演唱会进行页开始，不包含自动启动游戏与自动选曲。
- 9 轨为实验入口，仅执行预检，不发送正式动作。
- 判定点模板为 1280×720 自采样本；游戏界面大改后可能需要重新采集。

## 开发者

仓库已包含全部离线构建所需的归档件（Python embeddable、依赖 wheel、MFAAvalonia、MaaFramework DLL），克隆后可完全离线构建：

```powershell
# 1. 解压运行时与依赖（幂等，可重复执行）
powershell -ExecutionPolicy Bypass -File tools/bootstrap.ps1

# 2. 项目检查（含离线运行时与原生库校验）
runtime\python\python.exe -B tools\check_project.py --require-runtime

# 3. 单元测试（178 项）
runtime\python\python.exe -B -m unittest discover -s tests -p "test_*.py" -q

# 4. 构建发行包（输出 dist\MAES）
powershell -ExecutionPolicy Bypass -File tools/package.ps1
```

目录说明：

- `agent/`：Python 动作服务与打歌引擎（`agent/music/` 为视觉跟踪、输入执行与运行时）。
- `resource/base/`：Pipeline、判定点模板与 OCR 模型；`resource/bside/` 为 B 服覆盖层（当前未启用）。
- `tests/`：178 项单元与契约测试。
- `tools/`：bootstrap、项目检查、测试与打包脚本。
- `provenance/`：资源来源与许可记录。
- `docs/`：架构、使用与诊断说明。

## 数据与隐私

- 本项目不收集、不上传任何数据；全部用户状态仅保存在本机。
- 发布包默认关闭在线版本检查与自动更新。

## 致谢与免责

- 致谢与第三方许可详见 `THIRD_PARTY_NOTICES.md`。
- **免责声明**：本项目仅供学习与技术交流，免费开源，禁止商用售卖；使用自动化工具可能违反游戏用户协议并带来账号风险，使用后果由使用者自行承担；本项目与游戏官方无任何关联。

## 许可

项目代码采用 MIT 许可（见 `LICENSE.md`）；第三方组件许可见 `THIRD_PARTY_NOTICES.md`。
