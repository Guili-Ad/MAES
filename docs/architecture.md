# 架构说明

## 总览

MAES 由两大部分组成：

1. **资源层（ProjectInterface V2 / Pipeline）**：`interface.json` 定义任务与设备入口；`resource/base/pipeline/my_task.json` 定义打歌相关节点（校准、探针、打歌运行、失败分派、终端识别、判定点与候选识别）。
2. **Agent（Python）**：随 MFAAvalonia 启动的子进程，注册自定义动作与自定义识别；打歌的全部视觉跟踪与输入调度在 `agent/music/` 内完成。

## 打歌运行时数据流

```
截图(1280x720) -> 候选提取(ColorMatch/numpy provider) -> 轨道关联/身份识别
      -> 命中时刻预测 -> 调度(deadline) -> 输入执行器(Maatouch 触摸/划动)
      -> 用户数据：%LOCALAPPDATA%\MAES\calibration\music.json
```

- **校准**：`MusicCalibrate` 在演唱会进行页识别判定点（模板 `resource/base/image/music_lane_judge_point.png`），保存轨道折线、候选 ROI 与视觉基线。
- **跟踪**：`agent/music/tracking.py` 维护轨道生命周期（Tap/长按/划动/变轨路线），`holds.py`/`hold_policy.py`/`head_identity.py` 负责长按语义。
- **隔离边界**：`association.py`仅处理观测关联，`stationary.py`处理轨迹级静止证据，`hold_notes.py`规划长按小音符；引擎保留兼容编排。点按时序与长按时序仍使用各自策略，共享`motion.py`数值拟合与`components.py`像素原语，不跨模块修改对方截止时间。
- **统一物理点按**：`point_events.py`保存普通圆盘、绿星、黄色头和金圈的共同物理编号、事件生命周期和逐输入回执；所有实际点击均为Tap。四类视觉、样本要求和时序策略仍各自保留，95ms密集规则只用于原普通点按。`point_sources.py`仅在同轨至少三次同步中心及轮廓证据证明同一对象时合并识别入口，保留原事件编号与已发送事实；不按截止时间或归属合并。
- **白带关系**：`tap_hold_chain.py`仅保存旁路活动／休眠／关闭关系及金圈视觉观测，金圈执行不依赖头部成功、旧所有者存在或原头轨道。关系变化不关闭物理点按，不产生持续按压或额外划动；时间冻结不免除有效观测和新鲜度检查。旧`hold_marker_identity.py`和`sustain.py`持续分支仍保留。
- **近线恢复**：`gold_recovery.py`仅恢复已确认、健康、未开始输入的物理金圈；局部金色圆环与沙漏白翼证据拒绝固定判定环，保序一对一匹配不创建新身份。真正新像素在最多两个捕获帧、0.6秒内可恢复原身份，但陈旧预测仍不能执行。局部未知白带不增加所有权证据或终尾票。
- **半程头部恢复**：`head_recovery.py`只为已有至少三帧健康运动、未Down的普通／绿星／黄头，在预测位置的有界窗口直接读取本帧原始像素。先证明白核、家族圆盘及紧凑局部轮廓，再做原物理资格和尺寸检查；不放宽全局过滤。恢复与正常入口共用原轨迹、物理编号及事件，保序一对一分配；重复截图和静止HUD不生成运动，身份不明则保持未知。
- **独立划动资格**：`flick_eligibility.py`在出生、排队、等待和会话原生Down前要求同方向真实箭头、健康运动及最多两帧周期的有效观测预算（不超过0.6秒）。静止出生事实跨历史窗口保存；重复箭头像素不刷新运动年龄。可恢复撤销保留未发送身份，必须新正向箭头才能复活；Down尝试后绝不重发。兼容持有触点的长按尾释放不进入新资格分支。
- **共享像素性能**：连通域采用等价有序行扫描和在线聚合，按最早像素恢复输出顺序；固定金色HSV掩码使用穷举验证等价的整数代数。白带和金圈缓存仅保存有界、不可变采样几何，七路归属评分批处理。每帧重新读取像素，不跨帧复用识别结果、改变阈值或轨道生成公式。
- **排队资格**：`pending_eligibility.py`在主派发、截图前派发及划动期间派发前统一检查点按身份。健康遮挡不限定中轨；出生即静止的HUD不能接走真实移动音符。可恢复撤销保留物理编号，必须取得后来有效正向观测才复活，不靠旧预测或重复截图重发。已经取得的新画面在点按执行前完成一次更新；精确等待不得明知越过原观测年龄预算。
- **调度与执行**：`agent/music/runtime.py`主循环；`executor.py`通过MaaFramework同步直接动作执行。`flick_session.py`交错不同轨的到期独立划动，预分配触点，连续Down再交错Move/Up；deadline独立、不凑组、不并发。活动轨道锁阻止同轨Tap重叠，容量不足保留未开始事件，不抢持续触点。实际持有触点的旧尾划保持兼容路径。
- **诊断**：`tap_trace.py` 输出每轮 JSONL trace；`storage.py` 保存最近结果与触控状态。

## 关键模块

| 模块 | 职责 |
|---|---|
| `agent/actions/music.py` | 自定义动作：校准、探针、打歌入口、失败上报、终端识别 |
| `agent/music/runtime.py` | 主循环、门控、事件派发、结果记录 |
| `agent/music/vision.py` | 颜色家族分类、候选提供者（numpy/Maa） |
| `agent/music/tracking.py` | 轨道关联、运动拟合、长按/划动语义、路线（折线）管理 |
| `agent/music/executor.py` | 触点分配与触摸/划动执行、熔断与清理 |
| `agent/music/tap_*.py` | 点按身份、和弦、调度与遮挡恢复 |
| `agent/music/point_events.py` `/point_sources.py` | 四类物理点按共用生命周期、跨识别入口强证据去重 |
| `agent/music/calibration.py` `/storage.py` | 校准与用户状态存储 |

## 任务与节点

- `MusicCalibration`：校准任务（要求进行页 + 识别 7/9 判定点）。
- `MusicTouchProbe`：触控探针（暂停页诊断，不再作为打歌前置条件）。
- `MusicPlayStage` -> `MusicPlayRun7`：7 轨打歌（运行期以 LIVE 结算检测结束）。
- `MusicPlayStage9Experimental`：9 轨实验，仅预检。

## 版本与日志标识

- 构建身份由`build_identity.py`计算；包内`build-manifest.json`记录源码、Pipeline及实际依赖文件哈希。打包与解压核验全清单；启动只校验源码及Pipeline，避免重复扫描运行库。当前显示版本为`v1.0.6-residual-candidate`，不代替内容身份。
- 运行、恢复段、事件三级命名空间隔离去重；只有确认释放所有触点才能开始新段。未确认释放的触点不得再分配。
- 7轨入口保持`hold_notes_as_taps=true`和`hold_sustain_enabled=false`；旧持续动作留作兼容测试，不自动启用。
