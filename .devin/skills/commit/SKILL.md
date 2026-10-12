---
name: commit
description: Make one git commit that does exactly one thing, authored as blitzcolo, with a Conventional Commits message free of milestone/progress markers and session links. Use whenever work in this repo is ready to be committed.
allowed-tools:
  - read
  - grep
  - glob
---

# 一事一提交

规则全文在根目录 `AGENTS.md`「Git 规则」。本流程把它落成步骤。

## 1. 看清要提交什么

```bash
cd /mnt/d/QuantiloomFusion
git status --short
git diff --stat
git diff --cached --stat
git log --oneline -10
```

用一句话写出"这个 commit 做了什么"。写不成一句话（出现"并且/另外/顺便"）就说明不止一件事，要拆：

- 按文件拆：`git add <path> ...` 只暂存属于这件事的文件。
- 同一文件里混了两件事：先把其中一件的改动撤回到工作区之外（另存补丁或手动回退），提交一件后再恢复另一件。**不要用 `git add -p` / `-i`**（交互式，不可用）。
- 纯重构与行为改动、格式化与逻辑改动、无关文档修正，一律分开。
- 与这件事直接相关的测试和文档放在同一个 commit。

代码改动提交前必须已跑过 `cross-os-test`；datagen 纯重构还必须已跑过 `datagen-parity`。

## 2. 写 message

- Conventional Commits：`feat:` `fix:` `refactor:` `test:` `docs:` `chore:` `build:` `perf:`。
- 英文祈使句，subject ≤ 72 字符；正文（可选）写"为什么"，以及移植来源等事实。
- 禁止：里程碑/milestone、`M1`、阶段/Phase/Step N、任务编号（`T2.4`）、进度/`1/5`/`part 2 of 5`、`WIP`、任何 URL、会话/工具 trailer（`Session:`、`Generated-by:`、机器人 `Co-authored-by:`）。

提交前自检（把 message 存到临时文件再检查）：

```bash
MSG=$(mktemp)
cat > "$MSG" <<'EOF'
refactor: move path translation into quantiloom_fusion.core.paths
EOF
grep -nEi 'milestone|里程碑|\bM[0-9]+\b|phase *[0-9]|阶段|\bstep *[0-9]|\bT[0-9]+\.[0-9]+\b|进度|\b[0-9]+ */ *[0-9]+\b|part *[0-9]+ *of|\bWIP\b|https?://|session *(link|url|id)|^ *session:|会话|对话|generated-by|co-authored-by' "$MSG" && echo "FORBIDDEN PATTERN - rewrite the message" || echo "message ok"
```

## 3. 提交并核对身份

始终显式带身份（不修改 git config）：

```bash
git -c user.name=blitzcolo -c user.email=19224718+blitzcolo@users.noreply.github.com commit -F "$MSG"
git log -1 --format='%an <%ae> | %cn <%ce>%n%s'
rm -f "$MSG"
```

两个身份都必须是 `blitzcolo <19224718+blitzcolo@users.noreply.github.com>`。pre-commit 钩子改了文件导致失败时，暂存被改的文件后重新提交（新 commit，不 `--amend`）。

## 4. 不做的事

- 不 push、不 `--amend`、不 rebase、不改写历史，除非用户明确要求。
- 不提交 `build/`、`.venv*`、EXR、权重、密钥。
- 优先在 WSL 的 git 里提交；Windows git 的 `core.autocrlf=true` 由 `.gitattributes` 的 `eol=lf` 兜住，但不要两边混着暂存同一批改动。
