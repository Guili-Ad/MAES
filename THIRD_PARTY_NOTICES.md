# Third-party notices

MAES 自有代码与自采资源采用 MIT 许可（见 `LICENSE.md`）。

## 发行包（组合式 Windows GUI 分发包）包含

- **MaaFramework 5.12.2**（LGPL-3.0）：原生库位于 `vendor/maaframework/`，打包复制为 `MaaFramework-LICENSE.md`。上游源码：<https://github.com/MaaXYZ/MaaFramework>（tag v5.12.2）。
- **MFAAvalonia 2.12.2**（GNU GPL v3）：GUI 宿主；完整文本见 `vendor/MFAAvalonia-LICENSE.txt`，打包复制为 `MFAAvalonia-LICENSE.txt`。上游源码：<https://github.com/MaaXYZ/MFAAvalonia>（tag v2.12.2）。
- **Python 3.12.10 嵌入式发行版**（PSF License）：`runtime/python/LICENSE.txt`。
- **NumPy 2.2.6**（BSD-3-Clause）、**MaaAgentBinary 1.0.1**、**StrEnum 0.4.15**（MIT）、**maafw 5.12.2**（LGPL-3.0）：许可与元数据位于 `runtime/python/Lib/site-packages/*.dist-info/`。
- **PaddleOCR PP-OCRv4 检测/识别模型与字典**（Apache-2.0）：位于 `resource/base/model/ocr/`，完整许可见 `LICENSES/Apache-2.0.txt`，模型清单见该目录下的 `README.md`。

## 组合分发提示

分发包含 MFAAvalonia 的组合包时必须遵守 GPL v3，并提供对应源码获取方式（上游仓库即可）。若希望维持纯 MIT 交付，可只发布 MAES 项目资源（`agent/`、`resource/`、`interface.json`），由用户自行安装兼容的通用 Client。

## 素材来源

- 打歌判定点模板为 MAES 本机 1280×720 自采样本（记录见 `provenance/assets.json`）。
- 其余识别素材由本项目独立采集，或来自上述开源模型的公开来源。

## 致谢

开发过程中参考了 MMleo 项目的功能设计与页面流程，特此致谢；本项目代码与素材均为独立实现或自行采集。
