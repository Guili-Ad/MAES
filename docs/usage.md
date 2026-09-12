# 使用说明

## 环境准备

1. Windows 10/11 x64；安装 .NET Desktop Runtime 10 与 VC++ 2015-2022 Redistributable
   （可管理员运行 `DependencySetup_依赖库安装_win.bat` 自动安装）。
2. 模拟器设置为 **1280×720**（本项目的截图坐标系），推荐 MuMu；其他模拟器保持相同分辨率与 16:9。
3. 在 MFAAvalonia 中添加设备；确认“输入模式”为 **Default 或 Maatouch**。
   - 误设为 AdbShell 会导致点击延迟与长按失效（本项目不再自动降级）。
4. 游戏语言与渠道：官服资源（`resource/base`）。B 服覆盖层暂未启用。

## 打歌流程

1. 启动游戏，进入演唱会进行页（能看到判定点）。
2. 运行「打歌点位校准」：要求画面为 1280×720 且判定点清晰；识别出 7 个判定点后保存成功。
   - V4 校准按“7/9轨@1280×720”保存；旧版校准会被拒绝。
3. 运行「临时打歌」：从当前进行页直接进入实时跟踪，整首歌自动进行。
   - 结束时以结算 LIVE 检测为准；中途手动停止会记录为 cancelled。
4. 触控异常时：暂停歌曲后运行「打歌触控能力检测」做诊断；普通打歌无需预先运行探针。
5. 「临时打歌（9轨实验）」仅做预检与候选识别回归，不发送正式打歌动作。

## 用户数据

- 默认目录：`%LOCALAPPDATA%\MAES`
  - `calibration/music.json`：点位校准。
  - `music_touch.json`：触控探针状态（诊断用）。
  - `music_last_result.json`：最近一次打歌结果与性能指标。
  - `logs/`：MaaFramework 侧日志。
- 可通过环境变量 `MAES_DATA_DIR` 指定其他目录。

## 反馈问题时请附带

1. 游戏内成绩截图：Perfect/Great/Good/Bad/Miss、Max Combo、SUPPORT 次数；
2. 整段录屏（若方便）；
3. 运行目录下 `logs/`（含 `tap-traces`）与 `debug/on_error`（若有）；
4. `%LOCALAPPDATA%\MAES\music_last_result.json`；
5. 模拟器名称与版本、输入模式、分辨率、以及是否使用校准后的同一设备签名。
