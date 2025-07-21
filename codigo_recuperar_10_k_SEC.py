# -*- coding: utf-8 -*-
"""
@author: Patricio_Mery
"""

import sec_edgar_downloader

dl = sec_edgar_downloader.Downloader("TuNombreDeCompañia", "tu.email@dominio.com")

ticker = "TSLA" # puede ser cualquier ticket

dl.get("10-K", ticker, limit=None)  # Los descargara los reportes 10-K, para TSLA los disponibles que son 15

print("Descarga completada. Los archivos están en la carpeta 'C:/users/name_usuario/sec-edgar-filings")
