import psycopg2
import os

DB_HOST = "localhost"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASS = ""

def conectar_bd():
    """Establece la conexión a la base de datos."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        return conn
    except Exception as e:
        print(f" Error conectando a la BD: {e}")
        return None

def preparar_tabla(cursor):
    """Crea la tabla si no existe"""
    sql = """
    CREATE TABLE IF NOT EXISTS tabla_1 (
        id SERIAL PRIMARY KEY,
        contenido TEXT NOT NULL,
        fecha_ingesta TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    cursor.execute(sql)
    print("Tabla 'tabla_1' verificada o creada.")