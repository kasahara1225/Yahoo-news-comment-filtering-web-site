#!/usr/bin/env python
# coding: utf-8
import re
import time
import streamlit as st
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import joblib
import pandas as pd
from website import ArticleParser
from dotenv import load_dotenv


#行の表示数の上限を撤廃
pd.set_option('display.max_rows', None)

#列の表示数の上限を撤廃
pd.set_option('display.max_columns', None)


# print(sklearn.__version__)


# ↓↓↓↓テスト↓↓↓↓
# comments = ["やはり来月で3年目に差し掛かるこの侵攻を僅か24時間で終わらす事は不可能でしょうね。それは仕方が無いとしてこの次期大統 領が出来れば今年中にウクライナが一方的に不利な条件に立たされる中途半端な停戦ではなく、ロシアと言う侵略国家が2度とウクラ イナのみならず世界に牙を剥く事がない状態に立たされるような完全な終戦を期待しております"," アメリカがウクライナ戦争を停戦させるのは理論上は難しくない。"," 西側諸国は負けを認められないので時間がかかりそうです。"]
# article = "トランプ次期米大統領は7日の記者会見で、自らが実現を目指すロシアとウクライナの停戦が容易ではないとの認識をにじませた。これまでは「大統領就任前」や「就任後24時間以内」の停戦実現に意欲を示してきたが、今回は「（停戦まで）6カ月あれば良い」などと説明。プーチン露大統領との会談実現も、20日の就任以降になるとの見方を示し、目標を事実上後退させた格好だ。トランプ氏は会見で、停戦の実現について「6カ月あれば良い。それよりずっと前に解決できることを望む」と説明。プーチン氏との会談については、「プーチン氏は会いたいと思っているだろうが、20日以降でないと適切ではない」とした。その上で、「毎日多くの若者が殺されている」と述べ、早期停戦の必要性を改めて訴えた。トランプ氏はまた、ウクライナが求める北大西洋条約機構（NATO）への加盟に否定的な立場も改めて示した。「ロシアはプーチン氏が就任するずっと前から、NATOがウクライナに関わることはできないと言い続けてきた」と指摘。その上で、バイデン米大統領がウクライナの加盟の可能性に言及したとし、「ロシアの感情は理解できる」と主張した。一方、ロイター通信によると、トランプ次期政権のウクライナ・ロシア担当特使のキース・ケロッグ氏が、1月初旬に予定していたウクライナなどへの訪問をトランプ氏が就任する20日以降に延期したと報じた。停戦を巡っては、敵対する双方の主張に隔たりが大きく、仮にトランプ次期政権の仲介で停戦交渉が始まっても難航が予想されている。"


st.set_page_config(
    page_title="ヤフコメ表示サイト",
    page_icon="🍎",
    layout="wide",
)

st.markdown(
    """
    <style>
        /* ボタンマウスオーバーしたい*/
        .stButton > button {
            background-color: #645b7f;
            color: #afb7c2;
            border-radius: 8px;
            transition: background-color 1s ease;
        }
        .stButton > button:hover {
            background-color: #afb7c2;
            color: #645b7f;
            border: 2px solid #645b7f;
        }
        # /* スライダーの色 */
        # .stSlider > div > div > div > div[role="slider"]{
        #    background: linear-gradient(to right, #645b7f, #afb7c2);
        # }
        # /* スライダーのつまみ */
        # .stSlider > div > div > div > div > div[role="slider"]{
        #     background-color: #645b7f;
        #     box-shadow: 0px 0px 0px 0.2rem #afb7c2;
        #     color: #645b7f;
        # }
        /* コメント表示させるときフェードインとかしたい */
        .fade-in {
            animation: fadeIn 1.5s ease-in-out;
        }
        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }
        /* テキスト入力欄*/
        input::placeholder {
            color: #4a4a4a;
            font-weight: bold;
            border: #645b7f;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🍎 ヤフコメ表示サイト")
st.write("Yahoo!ニュースのコメントをランダムで15件表示します．")

url = st.text_input(
    "🔗 URLを入力してください", 
    placeholder="例: https://example~~~comments"
)
# threshold = st.slider("建設的度合いの閾値を設定してください", 0, 3, 1)



if st.button("💬 コメントを見る"):
    st.write("コメントを分析中です...")
    with st.spinner("分析中...お待ちください。"):
        time.sleep(7)

        
    parser = ArticleParser(url)
    get_contents = parser.get_comments_and_article()
    
    comments = get_contents.get("comments", [])
    article = get_contents.get("article", "")
    
    comments = [comment.replace("\n", "").strip() if isinstance(comment, str) else "不明なコメント" for comment in comments]


    if comments:
        st.write(f"{len(comments)} 件のコメントを取得しました。")
        comments_list = [
            (comment)
            for comment in zip(comments)
        ]
        comments_df = pd.DataFrame(comments_list, columns=["comment"])
        for i, row in enumerate(comments_df.itertuples(), 1):
            comment = row.comment
            st.write(f"""
            **{i}. {comment}**
            """)
 
# ↓↓↓↓テスト↓↓↓↓
# classifications = classify_comments(comments,article)
# predictions = classifications["prediction"]
# print("classifications↓")
# t = type(predictions)
# print(t) 
# print(predictions)

# attribute = attribute_with_gpt(comment,article)
# print(attribute)
# stance = stance_with_gpt(article,comment)
# print(stance)    
