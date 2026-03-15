# Medical RAG 评估脚手架 + 本地 RAG API（完全兼容）

本仓库同时提供：

1) **可重复、可追溯的评估脚手架**（对比 `rag` vs `ollama_llm_only` vs 可选 `openai_llm_only`）  
2) **本地 FastAPI RAG 服务**（与脚手架 **API 合同完全一致**，支持 HTTP 模式调用）

默认离线（Ollama），OpenAI 基线通过 `OPENAI_API_KEY` 开关启用。

---

## 目录结构（固定）

```
rag-eval-harness/
  README.md
  requirements.txt
  .gitignore
  .env.example

  src/
    __init__.py

    rag/
      __init__.py
      rag_adapter.py
      rag_config.py
      rag_api_client.py

    llm/
      __init__.py
      ollama_client.py
      openai_client.py

    eval/
      __init__.py
      schemas.py
      io_utils.py
      run_eval.py
      report.py

  configs/
    systems.yaml
    runs.yaml

  data/
    questions/
      README.md
      smoke_v1.jsonl
      full_v1.jsonl

  runs/
    .gitkeep

  scripts/
    make_questions_from_csv.py
    run_smoke.ps1
    run_full.ps1
    start_rag_api.ps1

  services/
    medical_rag_api/
      README.md
      requirements.txt
      run.ps1
      app/
        __init__.py
        server.py
        settings.py
        rag_backend.py
        schemas.py

  tests/
    test_questions_format.py
    test_smoke_run_dry.py
    test_rag_api_contract.py
```

---

## 1) 环境准备（Windows PowerShell）

在仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r services\medical_rag_api\requirements.txt
```

---

## 2) 放置你的 RAG（Original_Version2.0.py）

**最简单**：把你的 `Original_Version2.0.py` 复制到仓库根目录（与 README 同级）。

如果你想放在别处：
- 放到 `src/rag/vendor/Original_Version2.0.py` 也可以
- 然后设置环境变量 `USER_RAG_PATH` 或修改 `src/rag/rag_config.py`

⚠️ **重要**：如果你的 `Original_Version2.0.py` 里有交互循环（`input()` / while True），必须放进：

```python
if __name__ == "__main__":
    ...
```

否则 adapter import 时会阻塞。

---

## 3) 启动 RAG API 服务（HTTP 模式）

```powershell
scripts\start_rag_api.ps1
```

默认监听 `127.0.0.1:8008`

你可以用下面命令验证：

```powershell
Invoke-RestMethod http://127.0.0.1:8008/health
```

---

## 4) 运行 Smoke 评估（默认：rag + ollama_llm_only）

```powershell
scripts\run_smoke.ps1
```

输出会写入 `runs/<run_id>/`，包含：

- `config.json`
- `answers.jsonl`
- `metrics.csv`
- `report.md`

---

## 5) 运行 Full 评估（可选 OpenAI）

```powershell
scripts\run_full.ps1
```

若你设置了：

```powershell
$Env:OPENAI_API_KEY="sk-..."
```

则 `openai_llm_only` 会自动加入评估；否则自动跳过。

---

## 6) inproc vs http

默认推荐 `http`（隔离、可复现实验环境，顺带验证 API 合同）。

如需 **inproc**（不启动服务）：

```powershell
python -m src.eval.run_eval --run smoke --rag-mode inproc
```

---

## 常见问题

- 端口冲突：修改 `services/medical_rag_api/app/settings.py` 端口并同步 `configs/systems.yaml`
- HTTP 409：说明还没 index，返回 `{"error":"RAG_NOT_READY"}`（正常）

---

## 医疗免责声明

本项目仅用于研究/教学，不提供医疗建议。任何健康问题请咨询专业医生。
