
"""
Autor: Patricio Román Mery Araya. Se autoriza su uso bajo los términos y condiciones de la 
licencia Creative Commons Atribución-No Comercial 4.0 Internacional (CC BY-NC 4.0): 
https://creativecommons.org/licenses/by-nc/4.0/

Copia el Script en la carpeta 'C:/users/name_usuario/sec-edgar-filings' que es la que utiliza
SEC_Downloader al ejecutarlo se crearan las tablas

"""

import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time

CIK = "0001318605"  # CIK de TSLA
FORM_TYPE = "10-K"
USER_AGENT = "Tu_nombre (tu_email)"
HEADERS = {"User-Agent": USER_AGENT}
BASE_DIR = os.path.abspath("sec-edgar-tsla")
os.makedirs(BASE_DIR, exist_ok=True)

PALABRAS_CLAVE = [
    "balance",
    "statements of operations",
    "statements of income",
    "statements of cash flows",
    "statements of stockholders",
]

def tiene_palabra_clave(texto):
    texto = texto.lower()
    return any(palabra in texto for palabra in PALABRAS_CLAVE)

def get_filings_urls(cik, form_type, count=5):
    url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    resp = requests.get(url, headers=HEADERS)
    data = resp.json()
    filings = data['filings']['recent']
    urls = []
    for i, f_type in enumerate(filings['form']):
        if f_type == form_type:
            accession = filings['accessionNumber'][i].replace("-", "")
            doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{filings['primaryDocument'][i]}"
            urls.append(doc_url)
            if len(urls) >= count:
                break
    return urls

def descargar_y_extraer_tablas(url, carpeta_salida):
    html = requests.get(url, headers=HEADERS).text
    soup = BeautifulSoup(html, 'lxml')
    file_name = os.path.basename(url)
    filing_id = file_name.split('.')[0]
    os.makedirs(carpeta_salida, exist_ok=True)

    tablas = soup.find_all("table")
    tablas_guardadas = 0

    for i, tabla in enumerate(tablas):
        texto_tabla = tabla.get_text(separator=" ", strip=True).lower()
        if tiene_palabra_clave(texto_tabla):
            filas = []
            for fila in tabla.find_all("tr"):
                celdas = fila.find_all(["td", "th"])
                contenido_fila = [celda.get_text(strip=True) for celda in celdas]
                if contenido_fila:
                    filas.append(contenido_fila)
            if len(filas) >= 2:
                df = pd.DataFrame(filas)
                nombre_csv = f"{filing_id}_tabla_{i+1}.csv"
                ruta_csv = os.path.join(carpeta_salida, nombre_csv)
                df.to_csv(ruta_csv, index=False)
                print(f" Tabla guardada: {nombre_csv}")
                tablas_guardadas += 1

    return tablas_guardadas

# ======= EJECUCIÓN =======
urls = get_filings_urls(CIK, FORM_TYPE, count=5)
print(f" Se encontraron {len(urls)} URLs de 10-K para CIK.")

total_tablas = 0
for url in urls:
    print(f"\n Procesando: {url}")
    extraidas = descargar_y_extraer_tablas(url, BASE_DIR)
    print(f"  Tablas extraídas: {extraidas}")
    total_tablas += extraidas
    time.sleep(1.5)

print(f"\n Total de tablas clave extraídas: {total_tablas}")
print(f" Guardadas en: {BASE_DIR}")
