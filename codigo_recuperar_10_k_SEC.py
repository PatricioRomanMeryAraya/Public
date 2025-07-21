
"""
Autor: Patricio Román Mery Araya. Se autoriza su uso bajo los términos y condiciones de la 
licencia Creative Commons Atribución-No Comercial 4.0 Internacional (CC BY-NC 4.0): 
https://creativecommons.org/licenses/by-nc/4.0/

Copia el Script en la carpeta 'C:/users/name_usuario/sec-edgar-filings' que es la que utiliza
SEC_Downloader al ejecutarlo se crearan las tablas

"""

import os
import re
import pandas as pd
from bs4 import BeautifulSoup
import sec_edgar_downloader

TICKER = "TSLA"
DOC_TYPE = "10-K"
USER_AGENT_NAME = "TuNombreDeCompañia"
USER_AGENT_EMAIL = "tu.email@dominio.com"

print(" Descargando reportes 10-K desde EDGAR...")
dl = sec_edgar_downloader.Downloader(USER_AGENT_NAME, USER_AGENT_EMAIL)
dl.get(DOC_TYPE, TICKER, limit=None)
print(" Descarga completada.")

BASE_FOLDER = os.path.join(os.path.expanduser("~"), "sec-edgar-filings", TICKER, DOC_TYPE)
OUTPUT_FOLDER = os.path.join(BASE_FOLDER, "extracted_tables_csv")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

TITULOS_CLAVE = [
    "consolidated balance sheets",
    "consolidated statements of operations",
    "consolidated statements of income",
    "consolidated statements of cash flows",
    "consolidated statements of stockholders",
]

def normalizar_texto(text):
    return re.sub(r'\s+', ' ', text).strip().lower()

def contiene_palabra_clave(text):
    norm = normalizar_texto(text)
    return any(titulo in norm for titulo in TITULOS_CLAVE)

def extraer_tablas_relevantes(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        soup = BeautifulSoup(f, 'lxml')

    tablas_relevantes = []
    elementos = soup.find_all(['p', 'div', 'center', 'b', 'strong', 'table'])
    titulo_actual = None

    for el in elementos:
        if el.name != 'table':
            texto = el.get_text(separator=' ', strip=True)
            if contiene_palabra_clave(texto):
                titulo_actual = texto
        else:
            if titulo_actual:
                filas = []
                for row in el.find_all('tr'):
                    celdas = row.find_all(['td', 'th'])
                    fila = [c.get_text(strip=True) for c in celdas]
                    if fila:
                        filas.append(fila)
                if len(filas) >= 2:
                    df = pd.DataFrame(filas)
                    tablas_relevantes.append((titulo_actual, df))
                titulo_actual = None
    return tablas_relevantes

print(" Buscando archivos HTML con tablas clave...")

txt_files_found = 0
tablas_totales = 0

for root, dirs, files in os.walk(BASE_FOLDER):
    for file in files:
        if file.endswith(".html") and "primary-document" in file:
            txt_files_found += 1
            file_path = os.path.join(root, file)
            print(f"\n Procesando: {file}")

            try:
                tablas = extraer_tablas_relevantes(file_path)
                if not tablas:
                    print(" No se encontraron tablas financieras clave.")
                    continue

                base_name = os.path.splitext(file)[0]

                for idx, (titulo, df) in enumerate(tablas):
                    nombre_limpio = re.sub(r'[^a-zA-Z0-9_]', '', titulo.replace(' ', '_').lower())
                    csv_name = f'{base_name}_{nombre_limpio}_table_{idx+1}.csv'
                    csv_path = os.path.join(OUTPUT_FOLDER, csv_name)
                    df.to_csv(csv_path, index=False)
                    print(f" Tabla clave guardada: {titulo}")
                    tablas_totales += 1

            except Exception as e:
                print(f" Error: {e}")

print(f"\n Proceso finalizado.")
print(f" Archivos procesados: {txt_files_found}")
print(f" Tablas clave extraídas: {tablas_totales}")
print(f" Guardadas en: {OUTPUT_FOLDER}")
