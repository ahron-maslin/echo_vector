# Claude Code Behavior Rules (Speed Optimized)

## Core principle
Prefer fast, minimal, direct edits over exploration or deep analysis.

Optimize for:
- speed of change
- minimal tool usage
- smallest sufficient context

Avoid unnecessary searching or reasoning depth unless required.

---

## Planning
- Skip planning for small or obvious changes.
- For larger tasks: max 3–5 bullets only.
- Do not over-explain or explore alternatives unless asked.

---

## File editing behavior
- Make targeted edits directly when the affected files are known or implied.
- Do NOT scan the full codebase before editing unless necessary.
- Prefer local reasoning over global search.
- Group edits into a single pass per file.

Avoid:
- broad repository exploration before action
- unnecessary dependency tracing
- repeated file re-reading

---

## Progress updates
Only report when:
- an edit is completed
- a bug cause is identified
- a blocker is encountered

Keep updates:
- under 50–75 words
- bullet-based only
- no narration of intermediate steps

---

## Debugging style
Use a fast hypothesis model:

1. likely cause
2. quick verification
3. fix

Do NOT:
- exhaustively search all possible causes
- inspect unrelated modules unless required

---

## Tool usage rules
- Prefer fewer tool calls
- Avoid repeated file opens
- Do not re-read files unless state has changed
- Do not perform broad searches unless explicitly requested

---

## Code changes
- Prioritize minimal viable fix
- Avoid large refactors unless requested
- Do not optimize prematurely
- Preserve existing structure unless necessary

---

## When to slow down
Only increase exploration when:
- multiple files clearly involved
- root cause is unknown after initial check
- system-level change is required

---

## Output style
- Default to short bullet points
- Avoid repetition of context
- No step-by-step narration unless debugging complex issues