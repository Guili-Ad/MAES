# 诊断说明

## 日志与文件

| 文件 | 内容 |
|---|---|
| `logs/log-*.log` | MFAAvalonia/Agent 汇总日志：任务开始结束、失败码、性能告警 |
| `logs/tap-traces/*.jsonl` | 引擎 trace：候选/调度/输入/长按/划动/恢复等逐条记录（含版本与配置头） |
| `debug/maafw.log` | MaaFramework 原生日志 |
| `debug/on_error/*.png` | 失败瞬间截图 |
| `%LOCALAPPDATA%\MAES\music_last_result.json` | 最近打歌结果：状态、失败码、provider、输入模式、性能指标 |
| `%LOCALAPPDATA%\MAES\music_touch.json` | 触控探针结果（诊断） |

## 常见问题定位

- **“V4 校准无效”**：尚未校准或校准数据被清除。先运行「打歌点位校准」。
- **点击延迟 / 长按失效**：设备“输入模式”不是 Default/Maatouch；或模拟器负载过高（查看 `loop P95`）。
- **打歌中途异常结束**：查看 `logs` 中的失败码与 `on_error` 截图；`cancelled` 表示手动停止。
- **暂停页识别失败**（探针在部分模拟器布局下）：探针仅诊断用途，不影响打歌；可忽略。
- **性能告警**：`Music runtime loop P95 ... exceeds` 表示主循环偏慢；关闭其他高负载程序或降低模拟器渲染压力。

## 复盘方法

1. 用 trace 头部 `version` 确认运行版本；
2. 在 `logs/log-*.log` 中检索 `Music head action trace ... entries=` 获取每个动作的 `late_ms`（迟到毫秒）；
3. 结合录屏对齐分数出现时刻，区分“迟到点按”与“识别丢失”（`unscheduled head lost`）；
4. 长按类问题关注：`hold start/tail acquired/release locked`、`hold end flick bound`、`isolated incoming head from active hold`。
