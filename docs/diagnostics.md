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

1. 用 trace 头部 `build_id`、`run_id`、`config_hash`、`calibration_hash`确认实际运行身份；不能只看version。`effective_calibration_hash`排除创建时间，只比较实际几何/视觉参数；
2. 在 `logs/log-*.log` 中检索 `Music head action trace ... entries=` 获取每个动作的 `late_ms`（迟到毫秒）；
3. 结合录屏对齐分数出现时刻，区分“迟到点按”与“识别丢失”（`unscheduled head lost`）；
4. 当前点按化长按优先复核schema 5的`gold_observation / gold_recovered / tap_hold_anchor`及`hold_note_*`输入、资格和归属；`hold start/tail acquired/release locked`、`hold end flick bound`仅用于仍保留的旧持续按压兼容分支，不能用旧释放状态解释当前金圈归属。

## 优化候选指标与限制

结果格式为schema 3，当前trace为schema 5（旧字段保留），旧格式仍可读取。最近60项是精确窗口，不等同于全曲；`whole_run`是1 ms分桶全曲统计，分位数是桶上界。超过5000 ms的样本报告overflow，分位数落入溢出桶时为null，不能解释为0。
`missing_source_times`记录来源未知的事件，它们不参与截图到输入的延迟统计。主机输入调用时间不是游戏判定时间，排队等待也不等于识别计算耗时。
trace在内存中有界保存，结束写盘；`dropped_records`不为0时必须承认记录不完整。
schema 4区分`critical_dropped`（关键事件截断）、`visual_dropped`（视觉采样记录截断）及`visual_sampled_out`（主动略过高频重复视觉记录）；两类缓冲隔离，视觉噪声不淘汰输入和身份变更。
长按小音符关注`hold_note_tap / hold_note_refine / hold_note_group / hold_note_cancelled / hold_note_input / hold_note_anchor_done`。实际输入携带`origin / owner / marker`、最新视觉时间及原始预测；负track编号仅供兼容诊断，不取普通点按的预测。已排队不是已成功输入，确认终端抬起完成才允许退休。

schema 5补充`gold_observation`（采样的物理圆环、拓扑、归属证据）、`gold_recovered`（原物理身份的当前局部圆环，含预测中心、残差、搜索范围和来源截图时间）、`tap_hold_anchor`、`tap_hold_route_evidence`、`tap_physical_alias`及`flick_started/deferred`。`input`的`origin=flick`含Down、逐Move、Up主机调用时间、错误、未确认清理与降级原因；不能将完整划动返回时刻当成起按时刻。只有尝试Down的事件消费去重身份，预分配但未Down的成员仍留队列。金圈归属变化不改变其原始事件编号；内部圈不会关闭虚拟长按，已确认真实终尾或已绑定尾划成功才触发关闭检查。

`hold_note_cancelled.qualification`明确区分所有权关闭／新头替代、运动不足、观测年龄／帧预算及矛盾证据。`visual_age/coast_budget`是秒；时间冻结不会忽略这些资格。恢复真像素与延长旧预测是两回事，已实际Down的物理事件保留去重记录，不因原owner被清理而重发。

`hold_note_reacquired`记录撤销后同一未发送身份的重获。它先用最新有限预测、既有链资格和当前截图来源重建deadline，再进入原20ms冻结；不能沿用已经失效的旧deadline立即重新冻结。已排队身份不套用新生音符的50ms过去／350ms未来入队范围，也不把过期deadline钳到now；`correction_late_ms`保留相对于最新预测的真实迟到。此记录是恢复输入资格，不是重发已经尝试Down的音符。

`identity_refresh`记录截图后、输入前的金圈资格预处理；不要只看随后`tracking`下降就声称CPU降低。开发循环回放schema 3的`full_tracking_timing_ms`包含`engine.update`和外部金圈预处理，内嵌refresh不双计；不包含provider、掩码、OCR、输入或release/refine队列规划。旧`timing_ms`保留原更新口径，模拟`metrics_ms`不是实测CPU。

开发回放支持`tap_replay.py --loop --cost-profile ... --config ...`，调用生产打歌循环并模拟截图、候选、跟踪、OCR和输入成本，可注入失败。配置可来自JSON或实战trace的JSONL头；未提供时明确警告默认配置不是实战配置。已知候选回放没有白色条带和连接弧线像素，视频回放使用压缩帧及NumPy；两者均不能证明实机Maa识别或游戏FC。超长长按须包含头部前导，否则缺失所属状态不能用于尾端验收。
当前测试账号没有Support转化技能，直接以Bad/Miss验收；有此技能的账号才另将救回Perfect按Miss记。记录须填写曲名、设置、构建编号和运行编号。
