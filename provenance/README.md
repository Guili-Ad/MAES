# 原型资源门禁

`assets.json` 记录所有尚未重新采集或仍需上游许可证复核的文件。

- `release_allowed: false` 表示仅可用于本地开发验证。
- 替换资源时需要重新记录文件的 SHA-256、采集日期、游戏渠道和验证分辨率。
- 只有来源和授权均确认、并完成官服/B服实机验证后，才能改为 `release_allowed: true`。
- `tools/check_project.py --release` 会阻止任何未批准资源进入正式包。

