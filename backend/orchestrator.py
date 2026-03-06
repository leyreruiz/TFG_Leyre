"""Orchestrator: clasifica la intención del estudiante y enruta al agente adecuado."""

from backend.models.schemas import StudentRequest


class Orchestrator:

    def __init__(self, agents):
        self.agents = agents

    def detect_intent(self, message: str) -> str:
        """Clasifica la intención del mensaje del estudiante.

        Por ahora usa reglas simples basadas en palabras clave.
        """
        message_lower = message.lower()

        if any(kw in message_lower for kw in ("examen", "exam", "test", "pregunta tipo")):
            return "exam"

        #if any(kw in message_lower for kw in ("plan", "planifica", "calendario", "horario")):
            #return "study_plan"

        if any(kw in message_lower for kw in ("resumen", "resume", "summary")):
            return "summary"
        
        return "summary"

    def route(self, message: str) -> dict:
        """Enruta el mensaje al agente correspondiente.

        1. Crea un StudentRequest
        2. Detecta la intención
        3. Busca un agente que pueda manejarla
        4. Devuelve la respuesta del agente
        """
        request = StudentRequest(message=message)
        request.intent = self.detect_intent(message)

        print(f"[Orchestrator] Intención detectada: {request.intent}")

        for agent in self.agents:
            if agent.can_handle(request.intent):
                print(f"[Orchestrator] Enrutando a: {agent.__class__.__name__}")
                return agent.handle(request)

        return {"error": f"No hay agente disponible para la intención '{request.intent}'"}
