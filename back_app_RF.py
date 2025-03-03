#!/usr/bin/env python
# coding: utf-8
# app_RF.py
import re
import time
import streamlit as st
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from xfeat import  Pipeline, SelectNumerical, ArithmeticCombinations
from website import ArticleParser
from Attri_Cos_Senti_Stance import (
    attribute_with_gpt,
    strength_with_gpt,
    calculate_cosine_similarity,
    stance_with_gpt,
    strength_article
)


#行列の表示数の上限をなくしている.デバッグ用
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


#モデル
model = joblib.load("comment_classifier.pkl")

# ↓↓↓↓テスト↓↓↓↓
# comments = ["やはり来月で3年目に差し掛かるこの侵攻を僅か24時間で終わらす事は不可能でしょうね。それは仕方が無いとしてこの次期大統 領が出来れば今年中にウクライナが一方的に不利な条件に立たされる中途半端な停戦ではなく、ロシアと言う侵略国家が2度とウクラ イナのみならず世界に牙を剥く事がない状態に立たされるような完全な終戦を期待しております"," アメリカがウクライナ戦争を停戦させるのは理論上は難しくない。"," 西側諸国は負けを認められないので時間がかかりそうです。"]
# article = "トランプ次期米大統領は7日の記者会見で、自らが実現を目指すロシアとウクライナの停戦が容易ではないとの認識をにじませた。これまでは「大統領就任前」や「就任後24時間以内」の停戦実現に意欲を示してきたが、今回は「（停戦まで）6カ月あれば良い」などと説明。プーチン露大統領との会談実現も、20日の就任以降になるとの見方を示し、目標を事実上後退させた格好だ。トランプ氏は会見で、停戦の実現について「6カ月あれば良い。それよりずっと前に解決できることを望む」と説明。プーチン氏との会談については、「プーチン氏は会いたいと思っているだろうが、20日以降でないと適切ではない」とした。その上で、「毎日多くの若者が殺されている」と述べ、早期停戦の必要性を改めて訴えた。トランプ氏はまた、ウクライナが求める北大西洋条約機構（NATO）への加盟に否定的な立場も改めて示した。「ロシアはプーチン氏が就任するずっと前から、NATOがウクライナに関わることはできないと言い続けてきた」と指摘。その上で、バイデン米大統領がウクライナの加盟の可能性に言及したとし、「ロシアの感情は理解できる」と主張した。一方、ロイター通信によると、トランプ次期政権のウクライナ・ロシア担当特使のキース・ケロッグ氏が、1月初旬に予定していたウクライナなどへの訪問をトランプ氏が就任する20日以降に延期したと報じた。停戦を巡っては、敵対する双方の主張に隔たりが大きく、仮にトランプ次期政権の仲介で停戦交渉が始まっても難航が予想されている。"


# # website.pyで記事アクセス＆コメント読み込み

def classify_comments(comments: list, article: str) -> pd.DataFrame:
    df = pd.DataFrame({'comment': comments})
    results = []
    for comment in comments:
        attribute = attribute_with_gpt(comment, article).replace(" ", "")
        strength = strength_with_gpt(comment)
        cosine_sim = calculate_cosine_similarity(comment, article)
        stance = stance_with_gpt(article, comment)
        article_strength = strength_article(article)


        new_row = {
            "comment": comment,
            "attribute": attribute,
            "strength": strength,
            "cos": cosine_sim,
            "stance": stance,
            "art_stren": article_strength
        }
        results.append(new_row)
    results_df = pd.DataFrame(results)         
    print("results_df")
    print(results_df)
    
    encoder = Pipeline(
        [
            SelectNumerical(),
            ArithmeticCombinations(
                input_cols=["art_stren", "strength"], 
                drop_origin=True, 
                operator="+", 
                r=2,
                output_suffix="_plus"
            ),
        ]
    )
    encoded_results = encoder.fit_transform(results_df)
    categories_stance = ["FAVOR", "AGAINST"]
    for category_stance in categories_stance:
        results_df[category_stance] = results_df["stance"].apply(
            lambda x: 1 if category_stance in x.split(",") else 0
        )   
        
    categories = ["意見", "根拠", "解決策", "経験談", "非建設"]
    for category in categories:
        results_df[category] = results_df["attribute"].apply(
            lambda x: 1 if re.search(fr"\b{category}\b", x) else 0
        )
        
    results = pd.concat([results_df[["strength", "cos","art_stren"]], results_df[categories],results_df[categories_stance],encoded_results], axis=1)
    print("results↓")
    print(results)
    
    final_results = results[
        ["strength", "cos", "art_stren", "意見", "根拠", "解決策", "経験談", "非建設","AGAINST", "FAVOR", "art_strenstrength_plus"]]

    predictions = model.predict(final_results)
    # final_results["prediction"] = predictions
    final_results.loc[:, "prediction"] = predictions
    return final_results
    

st.set_page_config(
    page_title="ヤフコメフィルタリングサイト",
    page_icon="📝",
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

st.title("📝 ヤフコメフィルタリングサイト")
st.write("閾値を調整して、建設的なコメントを抽出します。")
st.write("「建設的なコメント」とは．．．☆を必ず満たしていて，◆のうちいずれか1つを満たしているコメント")
st.write(f"""              
                - ☆誹謗中傷を含まず，記事に関連しているコメント
                - ◆自分の意見をもとに議論を促している
                - ◆客観的な根拠を提示している
                - ◆新しい解決策を提示している
                - ◆珍しい体験談を提示している
                """)

url = st.text_input(
    "🔗 URLを入力してください", 
    placeholder="例: https://example~~~/comments"
)
threshold = st.slider("建設的度合いの閾値を設定してください", 0, 3, 1)

col1,col2,col3,col4 = st.columns(4)
with col2:
    show_evidence = st.checkbox("根拠", value=True)
with col3:
    show_solution = st.checkbox("解決策", value=True)
with col4:
    show_experience = st.checkbox("経験談", value=True)
with col1:
    show_nonconstructive = st.checkbox("非建設的なコメントを心して見る...👀（閾値が1以上の時のみ選択できます）", value=False)


if "show_comments" not in st.session_state:
    st.session_state["show_comments"] = False
if "show_under_comments" not in st.session_state:
    st.session_state["show_under_comments"] = False

if st.button("💬 コメントを見る"):
    st.session_state["show_comments"] = True
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
        # コメント分類
        classifications = classify_comments(comments,article)
        predictions = classifications["prediction"]
        print("classifications↓↓")
        print(predictions)
        filtered_comments_list = [
            (comment, degree)
            for comment, degree in zip(comments, predictions)
            if degree >= threshold
        ]
        filtered_comments = pd.DataFrame(filtered_comments_list, columns=["comment","degree"])
        filtered_df_1 = classifications[classifications["prediction"] >= threshold]
        filtered_df = pd.concat([filtered_df_1,filtered_comments], axis=1)
        # st.dataframe(classifications)
        # print("filtered_df　columns↓↓")
        # print(filtered_df.columns)
        print("filtered_df↓↓")
        print(filtered_df)

        if show_nonconstructive:
            filtered_df = filtered_df[filtered_df["prediction"] == 0]
        if show_evidence:
            filtered_df = filtered_df[filtered_df["根拠"] == 1]
        if show_solution:
            filtered_df = filtered_df[filtered_df["解決策"] == 1]
        if show_experience:
            filtered_df = filtered_df[filtered_df["経験談"] == 1]
            
        filtered_df = filtered_df.sort_index().dropna()

         ## 閾値以上の結果表示
        if st.session_state["show_comments"]:
            if not filtered_df.empty:
                st.write(f"🔍 **閾値以上のコメント ({len(filtered_df)} 件)：**")
                st.write(f"""
                    **用語説明**
                    - 感情強度: コメントのネガティブ・ポジティブの度合い．[-3, -2, -1, 0, 1, 2, 3]で表される.
                    - 記事との関連度合い: コメントが記事とどの程度関連しているかを示す度合い．[0 ~ 1]で表される.
                    - コメントの属性: 属性の種類は"意見", "根拠", "解決策", "経験談", "非建設"がある．
                    - スタンス: 記事に対するコメントの賛否．"FAVOR（賛成）", "AGAINST（反対）"がある．
                    """)
                for i, row in enumerate(filtered_df.itertuples(), 1):
                    comment = row.comment
                    degree = row.prediction
                    strength = row.strength
                    cos = row.cos
            
                    # 属性をone-hotから抽出
                    attributes = [col for col in ["意見", "根拠", "解決策", "経験談", "非建設"] if getattr(row, col) == 1]
                    attributes_text = ", ".join(attributes) 
            
                    # スタンスをone-hotから抽出
                    stance = [col for col in ["FAVOR", "AGAINST"] if getattr(row, col) == 1]
                    stance_text = ", ".join(stance)
    
                    print("comment")
                    print(comment)
                    print("stance_text")
                    print(stance_text)
            
                    st.markdown(
                        f"""
                        <div style="background-color: #F6F7F8; padding: 10px; border-radius: 10px; margin-bottom: 10px;">
                            <b>{i + 1}. {comment}</b><br>
                            - 建設的度合い: {degree}<br>
                            - 感情強度: {strength}<br>
                            - 記事との関連度合い: {cos:.2f}<br>
                            - コメントの属性: {attributes_text}<br>
                            - スタンス: {stance_text}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.warning("🔎 閾値以上のコメントは見つかりませんでした")
            
        


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
