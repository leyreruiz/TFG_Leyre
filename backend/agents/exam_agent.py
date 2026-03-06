"""Exam Agent: genera preguntas de examen tipo test basadas en un archivo específico."""

import os
import re

from backend.agents.base_agent import BaseAgent
from backend.models.schemas import StudentRequest
from backend.clients.llm_client import chat_with_model


class ExamAgent(BaseAgent):
    """Genera preguntas de examen tipo test basadas en un archivo específico."""

    def __init__(self, llm_model="llama3.2", num_questions=3, data_dir="backend/data"):
        self.llm_model = llm_model
        self.num_questions = num_questions
        self.data_dir = data_dir

    def can_handle(self, intent: str) -> bool:
        return intent == "exam"

    def handle(self, request: StudentRequest) -> dict:
        """Genera un examen basado en el archivo especificado.
        
        Args:
            request.message: Mensaje con palabras clave + nombre del archivo
                           Ej: "examen redes_neuronales" o "test bases_datos.txt"
        
        Returns:
            dict con content (texto raw) y questions (parseadas)
        """
        # Extraer el nombre del archivo del mensaje, ignorando palabras clave
        filename = self._extract_filename(request.message)
        
        if not filename:
            return {
                "agent": "exam",
                "error": f"No se especificó archivo. Uso: 'examen redes_neuronales' o 'exam bases_datos'. Disponibles: {self._list_available_files()}",
            }
        
        # Asegurar que tiene extensión .txt
        if not filename.endswith(".txt"):
            filename = filename + ".txt"
        
        # Construir la ruta del archivo
        filepath = os.path.join(self.data_dir, filename)
        
        # Verificar que el archivo existe
        if not os.path.exists(filepath):
            return {
                "agent": "exam",
                "error": f"Archivo no encontrado: {filename}. Disponibles: {self._list_available_files()}",
            }
        
        # Leer el contenido del archivo
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            return {
                "agent": "exam",
                "error": f"Error leyendo archivo: {e}",
            }
        
        if not content.strip():
            return {
                "agent": "exam",
                "error": f"El archivo {filename} está vacío.",
            }
        
        # Generar preguntas basadas en el contenido completo
        raw_answer = self._generate_questions(content, filename)
        
        if raw_answer is None:
            return {
                "agent": "exam",
                "error": "No se pudo generar el examen.",
            }
        
        # Parsear las preguntas a formato estructurado
        questions = self._parse_questions(raw_answer)
        
        return {
            "agent": "exam",
            "content": raw_answer,
            "questions": questions,
            "num_questions": len(questions),
            "source_file": filename,
        }

    def _extract_filename(self, message: str) -> str | None:
        """Extrae el nombre del archivo del mensaje, ignorando palabras clave.
        
        Ejemplos:
          "examen redes_neuronales" → "redes_neuronales"
          "exam bases_datos.txt" → "bases_datos.txt"
          "test sistemas_operativos" → "sistemas_operativos"
          "pregunta tipo redes_neuronales" → "redes_neuronales"
        """
        # Palabras clave a ignorar
        keywords = ["examen", "exam", "test", "pregunta tipo", "pregunta", "tipo"]
        
        # Convertir a minúsculas para comparación
        message_lower = message.lower().strip()
        
        # Remover cada palabra clave del inicio del mensaje
        for kw in keywords:
            if message_lower.startswith(kw):
                # Remover la palabra clave y los espacios
                message = message[len(kw):].strip()
                message_lower = message.lower().strip()
        
        # El resto debería ser el nombre del archivo
        if message:
            return message
        return None

    def _list_available_files(self) -> str:
        """Lista los archivos disponibles en el directorio de datos."""
        try:
            files = [f for f in os.listdir(self.data_dir) if f.endswith(".txt")]
            return ", ".join(files)
        except:
            return "No se pudieron listar los archivos"

    def _generate_questions(self, content: str, filename: str) -> str | None:
        """Llama al LLM para generar preguntas tipo test basadas en el contenido del archivo."""

        sistema = (
            "Eres un profesor universitario experto en crear exámenes. "
            "Tu tarea es generar preguntas de tipo test (opción múltiple) "
            "basándote ÚNICAMENTE en el contenido proporcionado. "
            "Cada pregunta debe tener exactamente 4 opciones (a, b, c, d) "
            "y solo una respuesta correcta."
        )

        prompt_usuario = f"""Contenido del archivo '{filename}':
---
{content}
---

Genera exactamente {self.num_questions} preguntas de examen tipo test.

Para CADA pregunta usa este formato exacto:

PREGUNTA 1: [texto de la pregunta]
a) [opción a]
b) [opción b]
c) [opción c]
d) [opción d]
RESPUESTA: [letra correcta]
EXPLICACIÓN: [breve explicación de por qué es correcta]

Asegúrate de que:
- Las preguntas cubran diferentes aspectos del contenido
- Las opciones incorrectas sean plausibles pero claramente distinguibles
- Las explicaciones sean concisas y útiles para el aprendizaje"""

        return chat_with_model(
            messages=[
                {"role": "system", "content": sistema},
                {"role": "user", "content": prompt_usuario},
            ],
            model=self.llm_model,
            temperature=0.5,  # Menos creatividad para exámenes más precisos
        )

    def _parse_questions(self, raw_text: str) -> list[dict]:
        """Intenta parsear la respuesta del LLM a una lista estructurada.

        Usa regex para extraer cada pregunta con sus opciones y respuesta.
        Si falla el parseo, devuelve lista vacía (el texto raw sigue disponible).
        """
        questions = []

        # Patrón para encontrar cada bloque de pregunta
        pattern = (
            r"PREGUNTA\s*\d+\s*:\s*(.+?)\n"  # pregunta
            r"\s*a\)\s*(.+?)\n"                # opción a
            r"\s*b\)\s*(.+?)\n"                # opción b
            r"\s*c\)\s*(.+?)\n"                # opción c
            r"\s*d\)\s*(.+?)\n"                # opción d
            r"\s*RESPUESTA\s*:\s*([abcd])\s*\n?"   # respuesta correcta
            r"\s*EXPLICACI[ÓO]N\s*:\s*(.+?)(?=\n\s*PREGUNTA|\Z)"  # explicación
        )

        matches = re.findall(pattern, raw_text, re.DOTALL | re.IGNORECASE)

        for match in matches:
            q_text, opt_a, opt_b, opt_c, opt_d, correct, explanation = match
            questions.append({
                "question": q_text.strip(),
                "options": [
                    opt_a.strip(),
                    opt_b.strip(),
                    opt_c.strip(),
                    opt_d.strip(),
                ],
                "correct_answer": correct.strip().lower(),
                "explanation": explanation.strip(),
            })

        return questions
