import newspaper
import os
from requests_html import HTMLSession

session = HTMLSession()
url = "https://news.google.com/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGRqTVhZU0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US%3Aen"
r = session.get(url)

r.html.render(sleep=1, scrolldown=12)

articles = r.html.find("article")
newsLinks = []

for i in articles:
    try:
        newsitem = i.find("a", first=True)
        rawLink = newsitem.attrs["href"]
        fullLink = str("https://news.google.com" + rawLink[1:-1])
        newsLinks.append(fullLink)
    except:
        pass

print(len(newsLinks))
print(newsLinks[5:10])

item = 1

switch = 0

list_of_urls = [
    "https://news.google.com/articles/CAIiEOxZgN3Og2P3MbBc5R-44EwqGQgEKhAIACoHCAowwL2ICzCckocDMOeRtwc?uo=CAUiZ2h0dHBzOi8vd3d3LmZveG5ld3MuY29tL3Nwb3J0cy9uZmwtd2Vlay05LWNoaWVmcy1yZXR1cm4tYWN0aW9uLXBsYXlvZmYtY29udGVuZGVycy1oZWFkLXBpdm90YWwtbWF0Y2h1cHPSAQA&hl=en-US&gl=US&ceid=US%3Ae",
    "https://news.google.com/articles/CCAiC0ZBMVhYM3RmUTJFmAEB?hl=en-US&gl=US&ceid=US%3Ae",
    "https://news.google.com/articles/CAIiEPaUDkK-R-GHnhnfCM9h-mEqFggEKg4IACoGCAow5tYTMODEAjDyugQ?uo=CAUihgFodHRwczovL3d3dy5jYnNzcG9ydHMuY29tL25mbC9uZXdzL25mbC13ZWVrLTktb2Rkcy1waWNrcy1zY2hlZHVsZS1ob3ctdG8td2F0Y2gtc3RyZWFtaW5nLWV4cGVydC1waWNrcy1zdXJ2aXZvci1waWNrcy10ZWFzZXJzLWFuZC1tb3JlL9IBAA&hl=en-US&gl=US&ceid=US%3Ae",
]

os.chdir("articles/technology/training")
for url in newsLinks:
    if item <= 100 and switch == 0:
        try:
            file2write = open("training_tech" + str(item) + ".txt", "w")
            url_i = newspaper.Article(url=(url), language="en")
            url_i.download()
            url_i.parse()
            text = url_i.text
            if len(text) > 0:
                file2write.write(text)
                item = item + 1
                file2write.close()
            if item == 101:
                os.chdir("../testing")
                switch = 1
        except:
            pass

    elif item > 100 and switch == 1:
        try:
            file2write = open("TeTechnologyArticle" + str(item) + ".txt", "w")
            url_i = newspaper.Article(url=(url), language="en")
            url_i.download()
            url_i.parse()
            text = url_i.text
            if len(text) > 0:
                file2write.write(text)
                item = item + 1
                file2write.close()
        except:
            pass
    elif item >= 200:
        break
