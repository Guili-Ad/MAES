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
- **长按小音符**：当前点按化模式由`tap_hold_chain.py`保存真实头部输入回执创建的虚拟身份、独立物理金圈和活动／休眠／关闭状态；不读取旧持续按压的释放时间、路径或watchdog。`hold_note_events.py`按运行段与物理金圈去重，归属为可修正元数据；时间冻结不免除视觉资格检查。条带未知不等于反证，实际金色圆环可保留运动而不制造所有权证据。旧`hold_marker_identity.py`和`sustain.py`持续分支仍保留。
- **近线恢复**：`gold_recovery.py`仅恢复已确认、健康、未开始输入的物理金圈；局部金色圆环与小时玻璃白翼证据拒绝固定判定环，保序一对一匹配不创建新身份。真正新像素在最多两个捕获帧、0.6秒内可恢复原身份，但陈旧预测仍不能执行。局部未知白带不增加所有权证据或终尾票。
- **共享像素性能**：连通域采用等价有序行扫描和在线聚合，按最早像素恢复输出顺序；固定金色HSV掩码使用穷举验证等价的整数代数。白带和金圈缓存仅保存有界、不可变采样几何，七路归属评分批处理。每帧重新读取像素，不跨帧复用识别结果、改变阈值或轨道生成公式。
- **排队资格**：`pending_eligibility.py`在主派发、截图前派发及划动期间派发前统一检查普通点按所属轨迹；允许健康遮挡coast，不因预测暂缺或一次候选拒绝就取消。
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
| `agent/music/calibration.py` `/storage.py` | 校准与用户状态存储 |

## 任务与节点

- `MusicCalibration`：校准任务（要求进行页 + 识别 7/9 判定点）。
- `MusicTouchProbe`：触控探针（暂停页诊断，不再作为打歌前置条件）。
- `MusicPlayStage` -> `MusicPlayRun7`：7 轨打歌（运行期以 LIVE 结算检测结束）。
- `MusicPlayStage9Experimental`：9 轨实验，仅预检。

## 版本与日志标识

- 构建身份由`build_identity.py`计算；包内`build-manifest.json`记录源码、Pipeline及实际依赖文件哈希。打包与解压核验全清单；启动只校验源码及Pipeline，避免重复扫描运行库。当前显示版本为`v1.0.4-tapchain-candidate`，不代替内容身份。
- 运行、恢复段、事件三级命名空间隔离去重；只有确认释放所有触点才能开始新段。未确认释放的触点不得再分配。
- 7轨入口保持`hold_notes_as_taps=true`和`hold_sustain_enabled=false`；旧持续动作留作兼容测试，不自动启用。
