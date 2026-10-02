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
- **调度与执行**：`agent/music/runtime.py` 主循环；`executor.py` 通过 MaaFramework 直接动作（TouchDown/TouchMove/TouchUp/Swipe）执行；默认按 Maatouch 高级输入运行。
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

- 构建身份由`build_identity.py`计算；包内`build-manifest.json`记录源码、Pipeline及实际依赖文件哈希。打包与解压核验全清单；启动只校验源码及Pipeline，避免重复扫描运行库。显示版本为`v1.0.1-opt-candidate`，不代替内容身份。
- 运行、恢复段、事件三级命名空间隔离去重；只有确认释放所有触点才能开始新段。未确认释放的触点不得再分配。
- 7轨入口保持`hold_notes_as_taps=true`和`hold_sustain_enabled=false`；旧持续动作留作兼容测试，不自动启用。
