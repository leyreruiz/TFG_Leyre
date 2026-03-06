import ollama
from backend.mcp_server import consultar_documentos_locales

MODELO = "llama3.2"

def agente_mcp_chat(pregunta_usuario):
    print(f"\n--- Nueva consulta: {pregunta_usuario} ---")
    
    # 1. Definición de la herramienta para Ollama
    herramientas = [{
        'type': 'function',
        'function': {
            'name': 'consultar_documentos_locales',
            'description': 'Busca información en los documentos del TFG cargados en ChromaDB',
            'parameters': {
                'type': 'object',
                'properties': {
                    'pregunta': {
                        'type': 'string',
                        'description': 'La duda o concepto a buscar',
                    },
                },
                'required': ['pregunta'],
            },
        },
    }]

    # 2. Primera llamada al modelo para decidir si usa herramientas
    mensajes = [{'role': 'user', 'content': pregunta_usuario}]
    
    respuesta = ollama.chat(
        model=MODELO,
        messages=mensajes,
        tools=herramientas,
    )

    # 3. Verificar si el modelo quiere llamar a la herramienta MCP
    if respuesta.message.tool_calls:
        print("[AGENTE] Llama 3.2 ha decidido usar el servidor MCP...")
        
        for tool in respuesta.message.tool_calls:
            if tool.function.name == 'consultar_documentos_locales':
                # Ejecutar la herramienta del servidor MCP
                contexto = consultar_documentos_locales(**tool.function.arguments)
                
                # Añadir el resultado al historial para que el modelo genere la respuesta final
                mensajes.append(respuesta.message)
                mensajes.append({
                    'role': 'tool',
                    'content': contexto,
                })

        # 4. Generar respuesta final con el contexto recuperado
        respuesta_final = ollama.chat(model=MODELO, messages=mensajes)
        return respuesta_final.message.content
    
    else:
        print("[AGENTE] El modelo responde sin usar herramientas externas.")
        return respuesta.message.content

if __name__ == "__main__":
    while True:
        user_input = input("\nTú: ")
        if user_input.lower() in ["salir", "exit"]: break
        print(f"\nIA: {agente_mcp_chat(user_input)}")