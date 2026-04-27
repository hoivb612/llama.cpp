# teacher.cpp — Teacher/Student Model Comparison Tool

## Problem
Build a tool that runs the same SYSTEM/PROMPT script through two models (teacher and student), compares their JSON answers, and when they disagree, asks the teacher to explain the reasoning that leads to the correct answer. The output is a result JSON file capturing mismatches and teacher reasoning.

## Approach
Sequential 3-pass design using the existing llm-infer library (which has global state — one model at a time):

1. **Pass 1 — Teacher**: Load teacher model, run all script prompts, save each answer as JSON → `teacher_results[]`
2. **Pass 2 — Student**: Load student model, run same prompts, save answers → `student_results[]`
3. **Pass 3 — Teacher Analysis**: Re-load teacher model. For each mismatched answer, send a meta-prompt asking for reasoning → store reasoning
4. **Output**: Write result JSON file with mismatches: `{ input_prompt, student_answer, teacher_answer, teacher_reasoning }`

## Config JSON File Format
```json
{
  "teacher_model": "path/to/teacher.gguf",
  "student_model": "path/to/student.gguf",
  "script_file": "path/to/script.txt",
  "output_file": "results.json",
  "n_threads": 4,
  "n_ctx": 2048,
  "verbose": 0,
  "template_file": "",
  "stop_char": "}"
}
```

## Answer Comparison Logic
- Parse JSON output from both models: `{"answer": "b. ...", "justification": "..."}`
- Extract the answer letter (first non-whitespace char of the "answer" field)
- Compare letters: if different → mismatch
- Log full answer text for review

## Teacher Re-analysis Prompt (Pass 3)
For each mismatch, send to teacher:
```
The student was given this prompt: "<original_prompt>"
The student answered: "<student_answer>"
The correct answer is: "<teacher_answer>"
Explain the reasoning or heuristic that would have led the student to the correct answer.
```

## Result JSON Output
```json
{
  "config": { ... },
  "total_prompts": N,
  "mismatches": M,
  "results": [
    {
      "prompt_index": 1,
      "input_prompt": "...",
      "teacher_answer": "c. That is correct...",
      "student_answer": "b. You might be confused...",
      "teacher_reasoning": "The student should have..."
    }
  ]
}
```

## Todos

### 1. create-teacher-cpp
Create `examples/llm-infer/teacher/teacher.cpp` with:
- Config JSON parsing (nlohmann/json)
- Script parser (reuse from minslm-multi-og.cpp)
- 3-pass execution loop (teacher → student → teacher analysis)
- JSON answer extraction and letter comparison
- Result file generation

### 2. update-cmake
Add `teacher` target to `examples/llm-infer/CMakeLists.txt` with nlohmann include dir

### 3. build-verify
Build and verify compilation

## Meta Command Handling
- `/rewind [N]` — **must be executed** in both teacher and student passes. It alters the KV cache, changing conversation context for subsequent prompts. Both models must see identical KV state at each prompt.
- `/context` — no-op for comparison purposes (just prints stats). Skip silently.
- `quit()` — ends the pass.
- Only prompts that produce model output (not meta commands) are recorded for comparison.

## Notes
- Reuses the same SYSTEM/PROMPT script format as minslm-multi-og.cpp
- Chat template detection (3-level: API → jinja scan → ChatML fallback) carried over
- `stop_char = '}'` default for JSON responses
- llm-infer global state means we must `llm_terminate()` + `llm_initialize()` between model loads
- nlohmann/json already vendored at `examples/llm-infer/include/nlohmann/json.hpp`
