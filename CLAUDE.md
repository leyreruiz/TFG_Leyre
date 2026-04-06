# CLAUDE.md — TFG_Leyre

## Descripcion del proyecto

**TFG_Leyre** es un Trabajo de Fin de Grado universitario: un asistente educativo con IA que genera resúmenes y exámenes automáticamente a partir de documentos (PDF, PPTX, TXT). El estudiante sube un tema, el sistema lo procesa y le devuelve estructura, resúmenes por sección y preguntas tipo test.

**Principios de desarrollo:**
- El código debe ser **sencillo, claro y correcto** — es un proyecto universitario, no producción.
- No añadir complejidad innecesaria. Cada pieza debe ser fácil de explicar.
- El idioma principal del proyecto es **inglés** (prompts, UI, lógica de negocio).

---

## Arquitectura general

```
Frontend (HTML/JS/CSS — SPA)
    ↕ HTTP + SSE (streaming)
Backend (FastAPI)
    ↕
Agents (Structurer → Summary → Exam → Explainer)
    ↕                    ↕
ChromaDB (RAG)     Groq LLM API (llama-3.1-8b-instant)
    ↕
MCP Server (expone la base de conocimiento como herramientas)
```

El pipeline principal es **síncrono y secuencial**: estructura → resúmenes por sección → preguntas por sección. Los resultados se envían al frontend en tiempo real mediante **SSE (Server-Sent Events)**.

---

## Estructura de ficheros

```
TFG_Leyre/
├── frontend/
│   └── index.html              # SPA completa (vanilla JS/HTML/CSS)
├── backend/
│   ├── api.py                  # FastAPI — endpoints principales
│   ├── class_storage.py        # Persistencia JSON de clases generadas
│   ├── mcp_server.py           # Servidor MCP (herramientas de conocimiento)
│   ├── mcp_client.py           # Cliente MCP (lanza el servidor como subprocess)
│   ├── utils.py                # normalize_topic(), extract_search_term()
│   ├── agents/
│   │   ├── base_agent.py       # Clase base: can_handle() + handle()
│   │   ├── structurer_agent.py # Genera el índice/estructura del tema
│   │   ├── summary_agent.py    # Genera resúmenes por sección (RAG)
│   │   ├── exam_agent.py       # Genera preguntas tipo test
│   │   └── explainer_agent.py  # Responde preguntas del alumno (historial)
│   ├── clients/
│   │   ├── llm_client.py       # Wrapper Groq SDK
│   │   └── bbdd_client.py      # Wrapper ChromaDB
│   ├── rag/
│   │   ├── retriever.py        # ChromaDbRetriever (.search() / .search_by_source())
│   │   └── ingest_topics.py    # Ingesta de ficheros → chunks → ChromaDB
│   ├── models/
│   │   └── schemas.py          # Modelos Pydantic (requests y domain models)
│   └── data/
│       ├── *.txt / *.pdf / *.pptx  # Ficheros fuente subidos
│       └── classes/            # Clases guardadas como JSON
├── chroma_db/                  # Base de datos vectorial (persistente en disco)
├── .env                        # GROQ_API_KEY
└── pyproject.toml              # Dependencias (gestor: uv)
```

---

## Componentes clave

### Agentes
Cada agente hereda de `BaseAgent` y tiene dos métodos: `can_handle(intent)` y `handle(request)`.

| Agente | Responsabilidad | Temperatura |
|--------|----------------|-------------|
| `StructurerAgent` | Analiza el documento y genera un índice (`###` por sección) | — |
| `SummaryAgent` | Genera resúmenes pedagógicos usando RAG (ChromaDB + LLM) | 0.7 |
| `ExamAgent` | Genera preguntas tipo test (4 opciones, 1 correcta + explicación) | 0.5 |
| `ExplainerAgent` | Responde preguntas del alumno con historial de conversación | 0.6 |

### Endpoints principales (`backend/api.py`)
| Endpoint | Método | Función |
|----------|--------|---------|
| `/start` | POST | Lanza el pipeline completo (SSE streaming) o carga clase guardada |
| `/regenerate-summary` | POST | Regenera el resumen de una sección |
| `/regenerate-exam` | POST | Regenera preguntas de una sección |
| `/add-questions` | POST | Añade preguntas adicionales sin repetir las existentes |
| `/update-questions` | POST | Actualiza lista de preguntas (tras borrar alguna) |
| `/ask` | POST | Pregunta del alumno → ExplainerAgent |
| `/ingest-and-prepare` | POST | Sube y procesa un fichero |
| `/classes` | GET/DELETE | Lista o elimina clases guardadas |

### RAG e ingesta
- Chunks de **500 caracteres con 50 de solapamiento** para no partir frases.
- Soporta `.txt`, `.pdf` (PyMuPDF) y `.pptx` (python-pptx).
- ChromaDB: colecciones nombradas `topic_{nombre_normalizado}`.

### Almacenamiento de clases
- Las clases generadas se guardan en `backend/data/classes/{topic}.json`.
- Estructura: `{topic, structure, sections, sections_data: {section: {summary, questions}}, created_at, updated_at}`.
- Permite recargar una clase sin regenerar todo.

### MCP Server
- Expone herramientas: `search_knowledge_base`, `get_class_data`, `search_wikipedia`, etc.
- El ExplainerAgent usa `search_wikipedia` para enriquecer las respuestas.
- Arranca como subprocess a través de `MCPRetriever` en `mcp_client.py`.

---

## Cómo ejecutar

### Requisitos
- Python 3.12+
- Clave de API Groq en `.env`: `GROQ_API_KEY=...`

### Arrancar el servidor
```bash
uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
```

### Acceder a la aplicación
- **Frontend:** http://localhost:8000
- **Swagger UI:** http://localhost:8000/docs

### Ingestar ficheros de ejemplo
```bash
python -m backend.ingest_topics
```

### Instalar dependencias
```bash
uv sync
```

---

## Modelo de dominio

**Flujo de datos principal:**
1. El usuario sube un fichero → se ingesta en ChromaDB.
2. El usuario inicia el aprendizaje → pipeline SSE:
   - `StructurerAgent` → índice con secciones.
   - Por cada sección: `SummaryAgent` → resumen → guardado. `ExamAgent` → preguntas → guardado.
3. El usuario puede preguntar dudas → `ExplainerAgent` (usa resumen de sección + Wikipedia).
4. El usuario puede regenerar resúmenes o preguntas individualmente.

**ExplainerAgent — restricciones de preguntas:**
- Solo responde preguntas relacionadas con el tema de la sección activa.
- Mantiene historial de hasta 6 turnos (12 mensajes) por `conversation_id`.
- Máximo 200 conversaciones simultáneas.

---

## Decisiones de diseño relevantes

- **Sin LangGraph ni frameworks de agentes**: el pipeline se orquesta con generadores Python puros, más simple y transparente.
- **SSE en lugar de WebSockets**: más simple para streaming unidireccional (servidor → cliente).
- **JSON para persistencia**: suficiente para un proyecto universitario; no se necesita base de datos relacional.
- **Groq + llama-3.1-8b-instant**: modelo rápido y gratuito, adecuado para el contexto académico.
- **Sin autenticación**: el sistema es de un solo usuario en local.
