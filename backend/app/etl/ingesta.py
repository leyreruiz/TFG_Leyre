import os
from ..clients.bbdd_client import conectar_bd, preparar_tabla
ARCHIVO_TXT = "datos.txt"


def ingestar_archivo(conn):
    """Lee el TXT e inserta los datos."""
    if not os.path.exists(ARCHIVO_TXT):
        print(f" El archivo {ARCHIVO_TXT} no existe.")
        return

    cursor = conn.cursor()
    
    # 1. Preparar la tabla
    preparar_tabla(cursor)
    
    print(f"Abriendo {ARCHIVO_TXT}...")
    
    # 2. Leer archivo
    cont_lineas = 0
    with open(ARCHIVO_TXT, 'r', encoding='utf-8') as f:
        for linea in f:
            linea_limpia = linea.strip()
            
            if linea_limpia:
                cursor.execute(
                    "INSERT INTO tabla_1 (contenido) VALUES (%s)", 
                    (linea_limpia,)
                )
                cont_lineas += 1

    # 3. Hacer commit para guardar los cambios
    conn.commit()
    cursor.close()
    print(f" Éxito: Se han ingestado {cont_lineas} líneas en la base de datos.")

if __name__ == "__main__":
    conexion = conectar_bd()
    if conexion:
        ingestar_archivo(conexion)
        conexion.close()
        print("Conexión finalizada.")