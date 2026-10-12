---
name: progress-update
description: Update docs/work-guide.md (progress table, decision log, open questions, handoff note) after finishing or pausing a task, and commit it separately. Use at the end of every task and before handing work over.
allowed-tools:
  - read
  - grep
  - glob
---

# 更新干活指引与进度备忘录

`docs/work-guide.md` 是下一个接手的人（或 agent）唯一的起点。它和代码不一致，比没有它更糟。

## 要改的地方

1. 顶部"最后更新"日期。
2. **§6 进度表**：任务状态（未开始 / 进行中 / 完成 / 搁置），备注里写关键事实（例如"对拍 PARITY OK，smoke 配置"），不要写"差不多""基本完成"。
3. **§7 决策记录**：本次做出的、以后会被问"为什么这样"的选择——日期、决策、理由。用户做的决定写"用户确认"。
4. **§8 待决问题与风险**：解决的移除或标注结论；新发现的加入，写清证据。
5. **§2 当前状态**：如果事实变了（新数据集目录、测试数、版本号、吞吐实测），同步更新，并写明来源（命令、目录、版本）。
6. **§9 交接备忘**：重写"当前停在"和"下一步"两行，具体到文件或命令。
7. 若任务的完成判据、接口形状（§3.3）或顺序变了，改对应章节，而不是只在进度表里补一句。

## 纪律

- 数字必须是实测并注明来源；推断标"待核实"。
- 任务编号只出现在本文里，**不能**出现在 commit message 中。
- 本文更新单独提交（用 `commit` skill），例如：
  `docs: note that path translation now lives in core.paths`。
  不要写成 `docs: update progress (T2.1 done)`。
