# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 13:21:06 2022

@author: AneeshDixit
"""

import requests


class Extraction:

    def __init__(self, query):
        self.query = query

        self.header = {'authority': 'www.amazon.in',
                       'method': 'GET',
                       'scheme': 'https',
                       'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
                       'accept-encoding': 'gzip, deflate, br',
                       'accept-language': 'en-US,en;q=0.9',
                       'sec-fetch-dest': 'document',
                       'sec-fetch-mode': 'navigate',
                       'sec-fetch-site': 'none',
                       'sec-fetch-user': '?1',
                       'sec-gpc': '1',
                       'service-worker-navigation-preload': 'true',
                       'upgrade-insecure-requests': '1',
                       'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36'
                       }

    def getSearch(self):
        url = "https://www.amazon.in/s?k=" + self.query
        print(url)

        page = requests.get(url, headers=self.header)

        if(page.status_code == 200):
            return page
        else:
            return "Error!!"

    def searchRevs(self, revLink):
        url = "https://www.amazon.in" + revLink
        page = requests.get(url, headers=self.header)

        if(page.status_code == 200):
            return page
        else:
            return "Error!!"

    def searchAsin(self, asin):
        url = "https://www.amazon.in/dp/"+asin
        page = requests.get(url, headers=self.header)
        if(page.status_code == 200):
            return page
        else:
            return "Error!!"
