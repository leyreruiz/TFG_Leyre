from sentence_transformers import SentenceTransformer

_modelo_cache = None

def obtener_modelo():
    """Obtiene el modelo en cache (cargado solo la primera vez)"""
    global _modelo_cache
    if _modelo_cache is None:
        print("Cargando modelo de IA (esto puede tardar un poco la primera vez)...")
        _modelo_cache = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return _modelo_cache

def generar_embedding(texto):
    """Genera un embedding para un texto"""
    modelo = obtener_modelo()
    return modelo.encode(texto).tolist()

def probar_embeddings():
    texto = "El profesor explica la teoría de la relatividad."
    vector = generar_embedding(texto)
    print(f" Texto convertido a vector.")
    print(f"Dimensiones del vector: {len(vector)}") 
    print(f"Primeros 5 números: {vector[:5]}") 

if __name__ == "__main__":
    probar_embeddings()