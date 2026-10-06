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
4. 当前候选trace为schema 6，保留schema 5旧字段；优先复核`gold_observation / gold_recovered / point_*`及`hold_note_*`输入、资格。`tap_hold_anchor`记录旁路白带关系，不是金圈执行门禁。`hold start/tail acquired/release locked`用于旧持续按压兼容分支，不能用旧释放状态解释当前金圈资格；尾划动同时查看独立的`flick_* / input`回执。

## 统一点按候选的事件链

v1.0.5仍兼容schema 5旧字段；`hold_note_*`仅表示金圈视觉来源，不表示需要持续触控或必须存在长按父身份。四类实际点击都补充`physical_id / visual_family / timing_profile`。

- `point_registered / point_identity_alias / point_source_authority`：注册、强证据身份合并、头部入口转为金圈入口；原事件编号与已经尝试的Down事实保持。
- `point_qualification`：包括尚未排队对象的拒绝原因，按物理来源和原因限频。金圈不再以“没有所有者”作为拒绝原因；有限采样日志的缺失不等于完全未识别。
- `point_cancelled / point_refresh_before_wait`：资格撤销及等待前请求刷新；不会扩大观测年龄预算或把截止改成当前时间。
- `point_input`：逐物理事件Down尝试、Up完成、异常；普通`input`仍保留完整主机调用回执，不能当作游戏已判Perfect。
- `point_identity_sent_conflict`：两个身份都已Down后发现冲突，保留两个输入事实，不静默合并来隐藏重复。

金圈Miss应按“有无正向观测→是否排队→是否Down尝试→是否Up完成→录像判定”区分；成功输入仍Miss必须独立核对实际像素到线区间、坐标及模型，不能用预测的到线时刻充当真值。

## 优化候选指标与限制

结果格式为schema 3，当前trace为schema 6（旧字段保留），旧格式仍可读取。最近60项是精确窗口，不等同于全曲；`whole_run`是1 ms分桶全曲统计，分位数是桶上界。超过5000 ms的样本报告overflow，分位数落入溢出桶时为null，不能解释为0。
`missing_source_times`记录来源未知的事件，它们不参与截图到输入的延迟统计。主机输入调用时间不是游戏判定时间，排队等待也不等于识别计算耗时。
trace在内存中有界保存，结束写盘；`dropped_records`不为0时必须承认记录不完整。
schema 4区分`critical_dropped`（关键事件截断）、`visual_dropped`（视觉采样记录截断）及`visual_sampled_out`（主动略过高频重复视觉记录）；两类缓冲隔离，视觉噪声不淘汰输入和身份变更。
长按小音符关注`hold_note_tap / hold_note_refine / hold_note_group / hold_note_cancelled / hold_note_input / hold_note_anchor_done`。实际输入携带`origin / owner / marker`、最新视觉时间及原始预测；负track编号仅供兼容诊断，不取普通点按的预测。已排队不是已成功输入，确认终端抬起完成才允许退休。

schema 5补充`gold_observation`（采样的物理圆环、拓扑、归属证据）、`gold_recovered`（原物理身份的当前局部圆环，含预测中心、残差、搜索范围和来源截图时间）、`tap_hold_anchor`、`tap_hold_route_evidence`、`tap_physical_alias`及`flick_started/deferred`。`input`的`origin=flick`含Down、逐Move、Up主机调用时间、错误、未确认清理与降级原因；不能将完整划动返回时刻当成起按时刻。只有尝试Down的事件消费去重身份，预分配但未Down的成员仍留队列。金圈归属变化不改变其原始事件编号；内部圈不会关闭虚拟长按，已确认真实终尾或已绑定尾划成功才触发关闭检查。

`hold_note_cancelled.qualification`在统一模式区分物理环未确认、出生即静止、运动不足、观测年龄及帧预算；所有权关闭／新头替代不再拒绝真金圈。旧日志保留`invalid-owner-or-marker`兼容标签，必须结合qualification读取，不能仅凭标签推断父链错误仍存在。`visual_age/coast_budget`是秒；时间冻结不会忽略这些资格。恢复真像素与延长旧预测是两回事，已实际Down的物理事件保留去重记录，不因原owner被清理而重发。

`hold_note_reacquired`记录撤销后同一未发送身份的重获。它先用最新有限预测、既有链资格和当前截图来源重建deadline，再进入原20ms冻结；不能沿用已经失效的旧deadline立即重新冻结。已排队身份不套用新生音符的50ms过去／350ms未来入队范围，也不把过期deadline钳到now；`correction_late_ms`保留相对于最新预测的真实迟到。此记录是恢复输入资格，不是重发已经尝试Down的音符。

`identity_refresh`记录截图后、输入前的金圈资格预处理；不要只看随后`tracking`下降就声称CPU降低。开发循环回放schema 3的`full_tracking_timing_ms`包含`engine.update`和外部金圈预处理，内嵌refresh不双计；不包含provider、掩码、OCR、输入或release/refine队列规划。旧`timing_ms`保留原更新口径，模拟`metrics_ms`不是实测CPU。

开发回放支持`tap_replay.py --loop --cost-profile ... --config ...`，调用生产打歌循环并模拟截图、候选、跟踪、OCR和输入成本，可注入失败。配置可来自JSON或实战trace的JSONL头；未提供时明确警告默认配置不是实战配置。已知候选回放没有白色条带和连接弧线像素，视频回放使用压缩帧及NumPy；两者均不能证明实机Maa识别或游戏FC。关系及尾划保护回放须包含头部前导；金圈物理点击本身不再要求头部或所有者存在。
当前测试账号没有Support转化技能，直接以Bad/Miss验收；有此技能的账号才另将救回Perfect按Miss记。记录须填写曲名、设置、构建编号和运行编号。

## 局部恢复与划动保护（schema 6）

- `head_recovery_search`：带原track、外观家族、首次发现、上次有效截图、预测中心与拒绝原因。它是限频的视觉记录，不能挤掉输入回执；`tap_mask_recovered`成功恢复记录保留，恢复不新增物理身份。
- `flick_qualification`：出生／排队／等待／Down前拒绝，区分静止出生、方向、运动不足、观测年龄／帧预算及等待新正向箭头。可恢复拒绝不是永久删除；静止出生不能因一次小跳位变成真箭头。
- `tap_dispatch_blocked`：划动期间到期点按因活动轨道、触点容量或单触兼容占用暂留队列，记录占用轨道、原deadline及组编号，不为绕开阻塞而改时。按事件、原因和恢复段限频，有界最多256个键。
- `input.first_seen_time`是物理来源首次发现；`latest_visual_time`和`latest_capture_started/finished`来自最近有效运动观测，不是事件排队时的旧截图。`visual_age_ms`是Down主机调用开始距该观测的年龄，`deadline_lateness_ms`是Down主机调用开始距最终deadline的偏差。Flick重复像素不更新这些有效运动时间。
- 旧`source_capture_*`和`perception_to_action / capture_to_action`保持事件来源口径及兼容，可能包含较长跟踪／等待，不能称为输入后端耗时。新增`latest_observation_to_action / latest_capture_to_action`单列最近观测／截图至主机Down的年龄，来源未知为null，排除于分位统计并计入`missing_latest_source_times`。
- `head_recovery`与`pause_prefilter`测量实际主机局部恢复与暂停预过滤耗时，使用原最近60项及全曲有界直方图。模拟回放的其他clock成本仍是注入值，不能与这些实际CPU墙钟混成实机循环性能。暂停预过滤阈值、OCR确认及严格结束识别不变。
