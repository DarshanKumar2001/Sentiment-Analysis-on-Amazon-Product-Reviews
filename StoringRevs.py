# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 13:28:08 2022

@author: AneeshDixit
"""

from bs4 import BeautifulSoup
from ExtractingRevs import Extraction
import pandas as pd



class Storing:

    def __init__(self, query):
        self.query = query

    def storingRevs(self):

        gettingRevs = Extraction(self.query)

        data_asin = []
        link = []
        reviews = []

        response = gettingRevs.getSearch()

        soup = BeautifulSoup(response.content, features="lxml")
        tags = {tag.name for tag in soup.find_all()}

        for tag in tags:
            for i in soup.find_all(tag):
                if i.has_attr("data-asin"):
                    if len(i['data-asin']) != 0:
                        data_asin.append("".join(i['data-asin']))

        for i in range(2, 4):
            response = gettingRevs.searchAsin(data_asin[i])
            soup = BeautifulSoup(response.content, features="lxml")
            for i in soup.findAll("a", {'data-hook': "see-all-reviews-link-foot"}):
                link.append(i['href'])

        for j in range(len(link)):
            for k in range(5):
                response = gettingRevs.searchRevs(
                    link[j]+'&pageNumber='+str(k))
                soup = BeautifulSoup(response.content, features="lxml")
                for i in soup.findAll("span", {'data-hook': "review-body"}):
                    reviews.append(i.text)

        rev = {'review_text': reviews}
        review_data = pd.DataFrame.from_dict(rev)

        return review_data, reviews
