# Informe Técnico y Funcional — TFG_Leyre

## 1. ¿Qué hace el proyecto?

Es un **asistente educativo con IA** que convierte documentos universitarios (PDF, PPTX, TXT) en material de estudio estructurado. El flujo desde el punto de vista del usuario es:

1. Sube un fichero (apuntes, presentación, etc.)
2. El sistema lo analiza y genera automáticamente un **índice de secciones**
3. Por cada sección genera un **resumen pedagógico**
4. Y unas **preguntas tipo test** para practicar
5. El alumno puede hacer preguntas sobre el tema y recibir respuestas del "profesor IA"

Todo ocurre en tiempo real: los resultados aparecen en pantalla conforme se generan, sin esperar a que todo termine.

---

## 2. Arquitectura general

```
┌──────────────────────────────────────────────────────────────┐
│  FRONTEND  (index.html — SPA en HTML/JS/CSS vanilla)         │
│  Se sirve desde el mismo servidor FastAPI                    │
└────────────────────────┬─────────────────────────────────────┘
                         │  HTTP + SSE (streaming en tiempo real)
┌────────────────────────▼─────────────────────────────────────┐
│  BACKEND  (FastAPI — api.py)                                 │
│  Orquesta el pipeline y expone los endpoints                 │
└──┬────────────────────────────────────────────────────┬──────┘
   │                                                    │
   ▼  Agentes Python                                    ▼  MCPRetriever
┌──────────────┐   ┌──────────────┐   ┌─────────────┐  ┌────────────────┐
│ Structurer   │   │ Summary      │   │ Exam        │  │ MCP Server     │
│ Agent        │   │ Agent        │   │ Agent       │  │ (subprocess)   │
└──────────────┘   └──────────────┘   └─────────────┘  └───────┬────────┘
         │                 │                  │                  │
         └─────────────────┴──────────────────┘                  │
                           │                                      │
              ┌────────────▼─────────────┐          ┌────────────▼─────────┐
              │  ChromaDB (vector DB)    │          │  Wikipedia API       │
              │  Embeddings por tema     │          │  (conocimiento ext.) │
              └──────────────────────────┘          └──────────────────────┘
                           │
              ┌────────────▼─────────────┐
              │  Groq API (LLM)          │
              │  llama-3.1-8b-instant    │
              └──────────────────────────┘
                           │
              ┌────────────▼─────────────┐
              │  JSON en disco           │
              │  backend/data/classes/   │
              └──────────────────────────┘
```

**Tecnologías clave:**
- **FastAPI**: framework web Python para los endpoints REST y streaming SSE
- **ChromaDB**: base de datos vectorial para búsqueda semántica (RAG)
- **Groq**: API que da acceso al LLM llama-3.1-8b-instant (rápido y gratuito)
- **MCP (Model Context Protocol)**: protocolo estándar para exponer herramientas al LLM

---

## 3. ¿Qué es RAG y por qué se usa?

**RAG (Retrieval-Augmented Generation)** es la técnica central del proyecto. El problema es que el LLM no "sabe" lo que hay en los apuntes del alumno. La solución es:

1. **Ingestar**: dividir el documento en trozos pequeños (chunks de 500 caracteres) y guardarlos en ChromaDB con sus **embeddings** (representaciones matemáticas del significado del texto)
2. **Recuperar**: cuando hay que generar un resumen de "Redes Neuronales", se buscan en ChromaDB los chunks más similares semánticamente a esa consulta
3. **Generar**: esos chunks se pasan al LLM como contexto para que genere una respuesta fundamentada en el documento real

Sin RAG, el LLM respondería con conocimiento general, no con el contenido específico de los apuntes.

---

## 4. El pipeline principal

Cuando el usuario hace clic en "Start", se ejecuta este pipeline:

```
PASO 1 — StructurerAgent
  Lee todos los chunks del documento → pide al LLM que genere un índice
  Resultado: "### 1. Introducción\n- Qué es X\n### 2. Métodos\n- Algoritmo A"

PASO 2 — Guardar esqueleto
  Se guarda inmediatamente un JSON vacío con la estructura, para que si
  algo falla a medias, el progreso no se pierda

PASO 3 — Por cada sección (en bucle):
  ├── SummaryAgent
  │     Busca en ChromaDB los 5 chunks más relevantes para esa sección
  │     Los pasa al LLM con el sub-índice de la sección
  │     Genera un resumen pedagógico estructurado
  │     Lo guarda en el JSON
  │
  └── ExamAgent
        Usa el resumen recién generado como contexto
        Genera N preguntas tipo test con 4 opciones y explicación
        Las guarda en el JSON

PASO 4 — Evento "done"
  El frontend sabe que todo ha terminado
```

Los resultados se envían al frontend mediante **SSE (Server-Sent Events)**: cada evento es un objeto JSON que el frontend recibe en tiempo real (como un stream de datos). El frontend no espera, va mostrando los resultados conforme llegan.

---

## 5. Descripción detallada de cada fichero

### `backend/api.py` — El núcleo del backend

Es el único fichero que "arranca" el servidor. Hace tres cosas:

**a) Inicializar los singletons** — Los agentes y el retriever se crean una sola vez al arrancar:
```python
_retriever  = MCPRetriever()        # Conecta con el MCP server
_structurer = StructurerAgent(...)  # Agente que genera el índice
_summary    = SummaryAgent(...)     # Agente que genera resúmenes
_exam       = ExamAgent(..., num_questions=2)  # Agente de examen
_explainer  = ExplainerAgent()      # Agente que responde preguntas
```

**b) Implementar las funciones del pipeline** — `_run_pipeline()` es un generador Python que va haciendo `yield _sse({...})`. Cada `yield` envía un evento al frontend. `_run_saved_pipeline()` hace lo mismo pero leyendo del JSON en lugar de llamar al LLM, lo que permite recargar una clase guardada sin coste.

**c) Definir los endpoints HTTP:**

| Endpoint | Qué hace |
|---|---|
| `POST /ingest-and-prepare` | Recibe el fichero, lo guarda, lo ingesta en ChromaDB |
| `POST /start` | Inicia la sesión (nueva o cargada del JSON) vía SSE |
| `POST /restart-with-suggestions` | Borra la clase y la regenera con feedback del usuario |
| `POST /regenerate-summary` | Regenera resumen y examen de una sección |
| `POST /regenerate-exam` | Regenera solo el examen de una sección |
| `POST /add-questions` | Añade preguntas sin repetir las existentes |
| `POST /submit-answer` | Guarda la respuesta del alumno a una pregunta |
| `POST /update-questions` | Guarda la lista tras borrar una pregunta |
| `POST /ask` | El alumno hace una pregunta → ExplainerAgent |
| `POST /clear-conversation` | Borra el historial del chat de una sección |
| `GET /topics` | Lista los ficheros disponibles en `backend/data/` |
| `GET /classes` | Lista las clases guardadas |
| `DELETE /classes/{topic}` | Borra clase JSON + fichero fuente + colección ChromaDB |
| `GET /` | Sirve el `index.html` |

---

### `backend/agents/` — Los cuatro agentes

**`base_agent.py`** — Clase abstracta con solo dos métodos: `can_handle(intent)` (¿puedo gestionar esta intención?) y `handle(request)` (hazlo). Todos los agentes la implementan.

---

**`structurer_agent.py`** — Genera el índice del documento

1. Extrae el nombre del tema del mensaje
2. Busca en ChromaDB todos los chunks del fichero (usando `search_by_source` si es un `.txt` conocido, o búsqueda semántica si no)
3. Pasa todo el contenido al LLM con este system prompt: *"Eres un profesor universitario creando un índice de clase"*
4. El LLM devuelve el índice en formato Markdown con `###` por sección
5. Si el usuario ha dado sugerencias (vía `/restart-with-suggestions`), se añaden al prompt

Temperatura: **0.4** — más baja = más determinista, índices más estables.

---

**`summary_agent.py`** — Genera resúmenes por sección

1. Busca en ChromaDB los **5 chunks más relevantes** para el título de la sección
2. Construye el prompt incluyendo:
   - El contexto recuperado (los chunks)
   - El sub-índice de esa sección (ej: los bullet points de "### 2. Backpropagation")
3. El LLM genera un resumen pedagógico siguiendo el orden del índice
4. El formato de salida incluye `##` por subsección, términos en **negrita**, analogías y un "Key Takeaway" al final

Temperatura: **0.7** — más alta = más creativo y explicativo.

---

**`exam_agent.py`** — Genera preguntas tipo test

Tiene dos modos de funcionamiento:

- **Pipeline mode**: cuando el mensaje contiene `"using the following summary context:"`, usa ese resumen directamente como contexto (evita volver a buscar en ChromaDB). Esto es lo que ocurre en el pipeline normal.
- **RAG mode**: búsqueda en ChromaDB por si se llama de forma independiente.

El parser `_parse_questions()` convierte el texto del LLM en una lista de dicts estructurados:
```json
{
  "question": "¿Qué es X?",
  "options": ["opción a", "opción b", "opción c", "opción d"],
  "correct_answer": "b",
  "explanation": "Porque..."
}
```

El método `add_questions()` lista las preguntas existentes en el prompt para que el LLM no las repita.

Temperatura: **0.5** — equilibrio entre variedad y precisión.

---

**`explainer_agent.py`** — Responde preguntas del alumno

Es el más complejo en cuanto a lógica de contexto. Para cada pregunta construye el mensaje así:

```
[system] Eres un profesor. Solo responde sobre el tema activo.
         Si la pregunta no es relevante, di [REFUSED] al final.
[system] Contexto de la sección: {resumen generado previamente}
[system] Contexto de la base de conocimiento: {chunks de ChromaDB}
[system] Contexto de Wikipedia: {artículo relacionado}
[historial] turno 1 usuario / turno 1 asistente / ...
[user]   {pregunta actual}
```

Gestiona un historial por `conversation_id` (máx. 6 turnos = 12 mensajes). Si hay más de 200 conversaciones activas simultáneas, elimina la más antigua.

El LLM debe incluir al final de cada respuesta `[Source: section]`, `[Source: wikipedia]` o `[Source: both]`. El agente extrae esa etiqueta antes de devolver la respuesta.

Si el LLM incluye `[REFUSED]`, la respuesta no se guarda en el historial.

Temperatura: **0.6**.

---

### `backend/mcp_server.py` + `backend/mcp_client.py` — El sistema MCP

Este es el componente más sofisticado del proyecto. MCP (Model Context Protocol) es un protocolo estándar creado por Anthropic para que los LLMs accedan a herramientas externas de forma estandarizada.

**¿Para qué se usa aquí?** Para exponer la base de conocimiento (ChromaDB + clases guardadas + Wikipedia) como "herramientas" con una interfaz estándar.

**`mcp_server.py`** — El servidor (corre como subproceso separado)

Define herramientas usando el decorador `@mcp.tool()`:

| Herramienta | Qué hace |
|---|---|
| `list_topics()` | Lista todos los temas disponibles |
| `search_knowledge_base(query, topic, k)` | Búsqueda semántica en ChromaDB |
| `get_class_data(topic)` | Devuelve el JSON completo de una clase |
| `get_section_summary(topic, section)` | Resumen de una sección específica |
| `get_section_questions(topic, section)` | Preguntas de una sección específica |
| `search_by_source(source, topic)` | Todos los chunks de un fichero concreto |
| `search_wikipedia(query, sentences)` | Busca en Wikipedia (primero ES, luego EN) |

Se arranca con `python -m backend.mcp_server`.

**`mcp_client.py`** — El cliente (vive dentro del proceso FastAPI)

La clase `MCPRetriever` es el puente entre los agentes y el MCP server:

1. Al instanciarse, **lanza el MCP server como subproceso** (`python -m backend.mcp_server`)
2. Se comunica con él por **stdio** (entrada/salida estándar) usando el protocolo MCP
3. Para los agentes, parece un retriever normal (mismo interfaz que `ChromaDbRetriever`): `.search()`, `.search_by_source()`, `.set_topic()`
4. Internamente, convierte cada llamada síncrona en una llamada asíncrona al subproceso

El truco técnico es que usa un **event loop en un hilo daemon separado** para no bloquear FastAPI. Las llamadas asíncronas al MCP server se despachan con `asyncio.run_coroutine_threadsafe()`.

**Flujo concreto de `search_wikipedia` en el endpoint `/ask`:**

```
api.py → MCPRetriever.search_wikipedia("redes neuronales")
       → asyncio.run_coroutine_threadsafe(_async_wikipedia(...))
       → MCP session.call_tool("search_wikipedia", {...})
       → [stdio] → mcp_server.py recibe la llamada
       → llama a wikipediaapi (primero ES, luego EN)
       → devuelve texto del artículo
       → [stdio] → MCPRetriever recibe la respuesta
       → api.py pasa el texto al ExplainerAgent como contexto
```

---

### `backend/ingest_topics.py` — Ingesta de documentos

Convierte ficheros en chunks y los guarda en ChromaDB:

1. **Extracción de texto** según el tipo de fichero:
   - `.txt`: lectura directa
   - `.pdf`: PyMuPDF extrae el texto de cada página
   - `.pptx`: python-pptx itera por diapositivas y formas, extrayendo texto y tablas

2. **Chunking**: divide el texto en trozos de **500 caracteres con 50 de solapamiento**. El solapamiento evita que una frase quede cortada entre dos chunks sin contexto.

3. **Guardado en ChromaDB**: cada chunk se guarda con metadatos `{topic, source, chunk_index}` en una colección llamada `topic_{nombre_normalizado}`.

---

### `backend/class_storage.py` — Persistencia JSON

Gestiona el almacenamiento de las clases generadas en `backend/data/classes/`. Cada clase es un fichero `{topic}.json` con esta estructura:

```json
{
  "topic": "neural_networks",
  "structure": "## Neural Networks\n### 1. Intro\n...",
  "sections": ["Intro", "Backpropagation", "..."],
  "sections_data": {
    "Intro": {
      "summary": "texto del resumen...",
      "questions": [
        {
          "question": "¿Qué es una red neuronal?",
          "options": ["...", "...", "...", "..."],
          "correct_answer": "a",
          "explanation": "...",
          "user_answer": "a",
          "user_correct": true
        }
      ],
      "conversation": [
        {"question": "¿puedes explicar X?", "answer": "Claro, X es..."}
      ]
    }
  },
  "created_at": "2026-04-06T12:00:00",
  "updated_at": "2026-04-06T12:05:00"
}
```

Funciones principales:
- `save_class()` — guarda la clase completa (preserva `created_at` si ya existía)
- `update_section_summary()` / `update_section_questions()` — actualiza una sola sección
- `append_section_conversation_turn()` — añade un turno al historial de chat
- `update_question_user_answer()` — guarda qué opción eligió el alumno y si acertó
- `delete_class()` — borra el JSON Y la colección ChromaDB asociada

---

### `backend/rag/retriever.py` — Retriever ChromaDB

Capa de abstracción sobre ChromaDB. Tiene una interfaz simple:

- `search(query, k)` — búsqueda semántica: ChromaDB convierte la query en un embedding y devuelve los k chunks más similares
- `search_by_source(fuente)` — recupera todos los chunks de un fichero específico, ordenados por índice

También existe `DummyRetriever` para tests (devuelve textos falsos sin tocar la BD).

---

### `backend/clients/`

**`llm_client.py`** — Wrapper mínimo del SDK de Groq. La función `chat_with_model(messages, model, temperature)` envía el historial de mensajes al LLM y devuelve la respuesta como string, o `None` si hay error.

**`bbdd_client.py`** — Wrapper de ChromaDB. Gestiona la conexión, la creación de colecciones por tema (`topic_{nombre}`), el guardado de chunks con embeddings (`save_text_chroma`), la búsqueda semántica (`search_similars`) y la búsqueda por metadatos (`search_by_source`).

---

### `backend/models/schemas.py` — Modelos Pydantic

Define los tipos de los datos que entran y salen de la API. Pydantic valida automáticamente que los campos existan y tengan el tipo correcto. Por ejemplo, si `/ask` recibe un JSON sin el campo `question`, FastAPI devuelve un error 422 automáticamente.

---

### `frontend/index.html` — La SPA completa

Es un fichero único que contiene HTML + CSS + JavaScript. No usa frameworks (React, Vue, etc.). Características notables:

- Escucha el stream SSE con `EventSource` y va renderizando cada evento conforme llega
- Distingue los eventos por su campo `type`: `"structure"`, `"summary"`, `"exam"`, `"done"`, `"error"`
- Gestiona el estado completo de la sesión en memoria (secciones, preguntas, chat)
- Hace fetch a los endpoints REST para operaciones puntuales (regenerar, añadir preguntas, etc.)

---

## 6. Flujo de datos completo — ejemplo real

**Escenario**: el alumno sube `redes_neuronales.pdf` y hace clic en "Start".

```
1. POST /ingest-and-prepare (multipart con el PDF)
   → Se guarda en backend/data/redes_neuronales.pdf
   → ingest_file_pdf() extrae el texto (PyMuPDF)
   → _divide_in_chunks() → ej: 47 chunks de 500 chars
   → save_text_chroma() × 47 → ChromaDB colección "topic_redes_neuronales"
   → Respuesta: {"topic": "redes_neuronales"}

2. POST /start {"document": "redes_neuronales"}
   → ¿Existe backend/data/classes/redes_neuronales.json? No
   → StreamingResponse(_run_pipeline("redes_neuronales"))

3. [SSE] _run_pipeline:
   → StructurerAgent.handle()
     → retriever.search("redes_neuronales", k=15) via MCP
     → MCP server: search_knowledge_base() → ChromaDB → 15 chunks
     → LLM genera el índice (temp=0.4)
   → yield {"type": "structure", "structure": "...", "sections": [...]}
   → save_class() → JSON esqueleto guardado

4. [SSE] Para cada sección (ej: "Backpropagation"):
   → SummaryAgent.summarize("Backpropagation", outline)
     → retriever.search("Backpropagation", k=5) via MCP
     → LLM genera resumen (temp=0.7)
   → update_section_summary() → JSON actualizado
   → yield {"type": "summary", "section_title": "Backpropagation", ...}

   → ExamAgent.handle("exam Backpropagation\nusing the following summary context:\n{resumen}")
     → LLM genera 2 preguntas (temp=0.5)
     → _parse_questions() → lista de dicts
   → update_section_questions() → JSON actualizado
   → yield {"type": "exam", "questions": [...]}

5. [SSE] yield {"type": "done"}

6. POST /ask {"question": "¿Por qué se llama backpropagation?", "section_title": "Backpropagation", ...}
   → retriever.search("¿Por qué se llama backpropagation?") via MCP → chunks relevantes
   → retriever.search_wikipedia("backpropagation") via MCP → artículo Wikipedia
   → ExplainerAgent.handle() con contexto: resumen + chunks + Wikipedia
   → LLM responde en conversación natural (temp=0.6)
   → append_section_conversation_turn() → historial guardado en JSON
   → Respuesta: {"agent": "explainer", "content": "..."}
```

---

## 7. Decisiones de diseño clave

| Decisión | Alternativa descartada | Por qué |
|---|---|---|
| Pipeline con generadores Python | LangGraph, LangChain | Más simple, más legible, sin dependencias pesadas |
| SSE para streaming | WebSockets | SSE es unidireccional (servidor → cliente), suficiente y más simple |
| JSON para persistencia | SQLite, PostgreSQL | Suficiente para un proyecto de un usuario, cero configuración |
| MCP para el retriever | Llamar ChromaDB directamente desde los agentes | Separa responsabilidades; el MCP server puede ser llamado por herramientas externas |
| Groq + llama-3.1-8b-instant | GPT-4, Claude | Gratuito, API key sin tarjeta, latencia muy baja |
| Sin autenticación | JWT, sesiones | Proyecto de un solo usuario en local |
| Chunks de 500 chars + 50 overlap | Chunks más grandes | Equilibrio entre contexto suficiente y precisión de recuperación |

---

## 8. Resumen en una frase por componente

| Fichero | Qué hace |
|---|---|
| `api.py` | Orquestador central: define el pipeline y todos los endpoints HTTP |
| `structurer_agent.py` | Lee el documento entero y genera el índice de secciones |
| `summary_agent.py` | Busca los trozos más relevantes del documento y los convierte en un resumen pedagógico |
| `exam_agent.py` | Convierte el resumen en preguntas tipo test con opciones y explicaciones |
| `explainer_agent.py` | Responde preguntas del alumno combinando resumen + ChromaDB + Wikipedia, con memoria de conversación |
| `mcp_server.py` | Expone ChromaDB y Wikipedia como herramientas con el protocolo estándar MCP |
| `mcp_client.py` | Lanza el MCP server como subproceso y lo usa desde el código Python síncrono |
| `ingest_topics.py` | Extrae texto de PDF/PPTX/TXT, lo trocea y lo guarda en ChromaDB |
| `class_storage.py` | Lee y escribe el JSON de cada clase generada |
| `bbdd_client.py` | Wrapper de ChromaDB: guardar, buscar por similitud, buscar por fuente |
| `llm_client.py` | Wrapper del SDK de Groq: una sola función `chat_with_model` |
| `schemas.py` | Define con Pydantic los tipos de los datos que entran y salen de la API |
| `retriever.py` | Abstracción sobre ChromaDB con interfaz `.search()` / `.search_by_source()` |
| `index.html` | La interfaz completa del alumno en un solo fichero HTML/JS/CSS |
