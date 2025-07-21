# -*- coding: utf-8 -*-
"""
@author: meryp
"""

import sec_edgar_downloader

dl = sec_edgar_downloader.Downloader("TuNombreDeCompañia", "tu.email@dominio.com")

ticker = "TSLA"

dl.get("10-K", ticker, limit=None)  # Los descargara todos los disponibles que son 15

print("Descarga completada. Los archivos están en la carpeta './sec-edgar-filings'.")