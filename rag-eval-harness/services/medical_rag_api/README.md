# Medical RAG API Service

FastAPI 服务（默认 `127.0.0.1:8008`），与评估脚手架完全兼容。

## 安装

在仓库根目录：

```powershell
pip install -r services\medical_rag_api\requirements.txt
```

## 启动

```powershell
scripts\start_rag_api.ps1
```

## 用 PowerShell 测试

### health

```powershell
Invoke-RestMethod http://127.0.0.1:8008/health
```

### index

```powershell
Invoke-RestMethod -Method POST http://127.0.0.1:8008/index `
  -ContentType "application/json" `
  -Body '{"data_sources":["data/documents"],"force_rebuild":false}'
```

### answer（如果没 index 会返回 409）

```powershell
Invoke-RestMethod -Method POST http://127.0.0.1:8008/answer `
  -ContentType "application/json" `
  -Body '{"question":"What is hypertension?","top_n":3,"score_threshold":0.0}'
```

医疗免责声明：仅研究/教学用途，不提供医疗建议。
