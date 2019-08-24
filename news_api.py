import requests
import math
import csv
import time

#necessity to get your own api key and put into an authentication file if you wish to run this code as is.
from authentication import api_key

data = requests.get("https://api.nytimes.com/svc/search/v2/articlesearch.json?q=amazon&page={0}&api-key={1}".format(0, api_key))

data_hits = data.json()["response"]["docs"][0]["web_url"][24:34]

page_number = 1

amazon_data = []
headline_date = []


#TODO
#check for file existence, and check for header presence
#write to different csvs and concatenate

while page_number <= 100:
    api_call = requests.get("https://api.nytimes.com/svc/search/v2/articlesearch.json?q=amazon&page={0}&api-key={1}".format((page_number + 10), api_key))
    for idx_key in range(len(api_call.json()["response"]["docs"])):
        amazon_data.append(api_call.json()["response"]["docs"][idx_key]["snippet"])
        headline_date.append(api_call.json()["response"]["docs"][idx_key]["web_url"][24:34])

    with open(f'amazon_data{page_number}.csv', 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["Headlines:", "Date:"])
        for idx in range(len(amazon_data)):
            wr.writerow([amazon_data[idx], headline_date[idx]])

    page_number += 1
    amazon_data = []
    headline_date = []
    time.sleep(10)

# print(headline_date, amazon_data)
