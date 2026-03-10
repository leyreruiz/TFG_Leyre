"""Orchestrator: classifies the student's intent and routes to the appropriate agent."""

from backend.models.schemas import StudentRequest


class Orchestrator:

    def __init__(self, agents):
        self.agents = agents

    def detect_intent(self, message: str) -> str:
        """Classify the student's message intent.

        Currently uses simple keyword-based rules.
        """
        message_lower = message.lower()

        if any(kw in message_lower for kw in ("examen", "exam", "test", "pregunta tipo")):
            return "exam"

        #if any(kw in message_lower for kw in ("plan", "planifica", "calendario", "horario")):
            #return "study_plan"  # future: study planner intent

        if any(kw in message_lower for kw in ("resumen", "resume", "summary")):
            return "summary"
        
        return "summary"

    def route(self, message: str) -> dict:
        """Route the message to the appropriate agent.

        1. Create a StudentRequest
        2. Detect the intent
        3. Find an agent that can handle it
        4. Return the agent's response
        """
        request = StudentRequest(message=message)
        request.intent = self.detect_intent(message)

        print(f"[Orchestrator] Detected intent: {request.intent}")

        for agent in self.agents:
            if agent.can_handle(request.intent):
                print(f"[Orchestrator] Routing to: {agent.__class__.__name__}")
                return agent.handle(request)

        return {"error": f"No agent available for intent '{request.intent}'"}
