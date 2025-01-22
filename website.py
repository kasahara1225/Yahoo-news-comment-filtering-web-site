#!/usr/bin/env python
# coding: utf-8

# # article.py

# In[28]:

from urllib.parse import urlparse
import random
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

COMMENT_TAG = "sc-169yn8p-10 hYFULX"
ARTICLE_TAG = "sc-54nboa-0 deLyrJ yjSlinkDirectlink highLightSearchTarget"
ANCHOR_PROPS_KEY = "data-cl-params"
ANCHOR_PROPS_VALUE = "_cl_vmodule:page;_cl_link:next;"

class ArticleParser(object):
    def __init__(self, root_url: str):
        self.root_url: str = root_url
        self.url = urlparse(root_url)
        self.title = ""

    def get_comments_and_article(self) -> dict:
        try:
            url: str = self.root_url
            all_comments: list[str] = []
            article_content: str = ""
            page_count = 0

            while page_count < 4:  # ページ遷移
                print(f"Fetching page {page_count + 1}: {url}")  # デバッグ

                response = requests.get(url)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "html.parser")
                # コメントを取得
                comments = soup.find_all("p", class_=COMMENT_TAG)
               
                if comments:
                    page_comments = [comment.get_text(strip=True) for comment in comments]
                    all_comments.extend(page_comments)
                    print(f"Found {len(comments)} comments on page {page_count + 1}")  # デバッグ
                else:
                    print(f"No comments found on page {page_count + 1}")  # デバッグ

                # 記事のリンクを取得
                article_anchor = soup.find(
                    "a", attrs={"data-cl-params": "_cl_vmodule:headline;_cl_link:title;"}
                )
                if article_anchor:
                    article_url = article_anchor.get("href")
                    print(f"Article URL found: {article_url}")  # デバッグ
                    article_response = requests.get(article_url)
                    article_response.raise_for_status()

                    article_soup = BeautifulSoup(article_response.text, "html.parser")
                    article_paragraph = article_soup.find("p", class_=ARTICLE_TAG)
                    if article_paragraph:
                        article_content = article_paragraph.get_text(strip=True)
                        print("Article content fetched successfully.")  # デバッグ
                    else:
                        print("Article content not found.")  # デバッグ
                else:
                    print("Article link not found.")  # デバッグ

                # 次のページへのリンクを取得
                next_anchor = soup.find(
                    "a", attrs={"data-cl-params": ANCHOR_PROPS_VALUE}
                )
                if next_anchor and next_anchor.get("href"):
                    # ホスト以降のリンクを結合して次のURLを取得
                    url = urljoin(url, next_anchor.get("href"))
                    page_count += 1
                else:
                    print("もうページが見つからない")  # デバッグ出力
                    break

            # コメントが見つからなかった場合の出力
            if not all_comments:
                print("すべてのページにコメントがなかった")  # デバッグ出力

            
            selected_comments = random.sample(all_comments, min(15, len(all_comments)))

            
            # 結果を辞書形式で返す
            return {"comments": selected_comments, "article": article_content}
        except Exception as e:
            print(f"An error occurred: {e}")  # エラーメッセージを出力
            return {}



# class ArticleParser(object):
#     def __init__(self, root_url: str):
#         self.root_url: str = root_url
#         self.url = urlparse(root_url)

#     def get_comments(self) -> list:
#         try:
#             url: str = self.root_url
#             comments: list[str] = []
#             page_count = 0

#             while page_count < 3:  # 最大3ページまで遷移
#                 response = requests.get(url)
#                 response.raise_for_status()
#                 soup = BeautifulSoup(response.text, "html.parser")

#                 # コメントを収集
#                 comment_elements = soup.find_all("p", class_=COMMENT_TAG)
#                 for comment in comment_elements:
#                     comments.append(comment.get_text(strip=True))

#                 # 次のページへのリンクを取得
#                 next_anchor = soup.find("a", attrs={ANCHOR_PROPS_KEY: ANCHOR_PROPS_VALUE})
#                 if next_anchor:
#                     # ホスト部分を結合して次のページのURLを作成
#                     url = self.url._replace(path=next_anchor.get("href")).geturl()
#                     page_count += 1
#                 else:
#                     break

#             return comments
#         except Exception as e:
#             print(f"Error occurred: {e}")
#             return []

