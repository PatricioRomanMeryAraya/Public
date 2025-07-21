
"""
Autor: Patricio Román Mery Araya. Se autoriza su uso bajo los términos y condiciones de la 
licencia Creative Commons Atribución-No Comercial 4.0 Internacional (CC BY-NC 4.0): 
https://creativecommons.org/licenses/by-nc/4.0/

Copia el Script a la carpeta en la carpeta 'C:/users/name_usuario/sec-edgar-filings' al 
ejecutarlo se crearan las tablas

"""

import os
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
print("Descarga completada.")

BASE_FOLDER = os.path.join(os.path.expanduser("~"), "sec-edgar-filings", TICKER, DOC_TYPE)
OUTPUT_FOLDER = os.path.join(BASE_FOLDER, "extracted_tables_csv")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def extract_tables_from_ixbrl(file_path):
    """Extrae tablas HTML desde archivo iXBRL"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        soup = BeautifulSoup(f, 'lxml')

    tables = soup.find_all('table')
    extracted_tables = []

    for table in tables:
        rows = []
        for row in table.find_all('tr'):
            cells = row.find_all(['td', 'th'])
            row_data = [cell.get_text(strip=True) for cell in cells]
            if row_data:
                rows.append(row_data)
        if rows:
            df = pd.DataFrame(rows)
            extracted_tables.append(df)

    return extracted_tables

print("Buscando archivos .txt con posibles tablas...")

txt_files_found = 0
txt_files_with_tables = 0

for root, dirs, files in os.walk(BASE_FOLDER):
    for file in files:
        if file.endswith(".txt"):
            txt_files_found += 1
            file_path = os.path.join(root, file)
            print(f"\n Analizando: {file}")

            try:
                tables = extract_tables_from_ixbrl(file_path)
                if not tables:
                    print("No se encontraron tablas HTML.")
                    continue

                txt_files_with_tables += 1
                base_name = os.path.splitext(file)[0]

                for idx, df in enumerate(tables):
                    csv_name = f'{base_name}_table_{idx+1}.csv'
                    csv_path = os.path.join(OUTPUT_FOLDER, csv_name)
                    df.to_csv(csv_path, index=False)
                    print(f"Tabla {idx+1} guardada ({len(df)} filas).")

            except Exception as e:
                print(f"Error procesando {file}: {e}")

print(f"\n Proceso terminado. Archivos .txt revisados: {txt_files_found}")
print(f" Archivos con tablas extraídas: {txt_files_with_tables}")
print(f" Archivos guardados en: {OUTPUT_FOLDER}")
