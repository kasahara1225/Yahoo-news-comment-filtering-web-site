#!/usr/bin/env python
# coding: utf-8
import os
from dotenv import load_dotenv
import pandas as pd
import re
import openai
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from streamlit import secrets


# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")
# FT_model = os.getenv("FT_model")

api_key = secrets["api_key"]
FT_model = secrets["FT_model"]
openai.api_key = api_key


def comment_strength(text):
    prompt = f"""
    この文章:「 {text}」について，感情極性の強度をそれぞれ-3～3で出力してください．
    強度の種類は「-3:強いネガティブ，-2:ネガティブ，-1:ややネガティブ，0:ニュートラル，1:ややポジティブ，2:ポジティブ，3:強いポジティブ」です．
    出力は「-3,-2, -1, 0, 1, 2,3」の数値のうちいずれか一つを出力してください．
    """
    response = openai.ChatCompletion.create(
        model=FT_model,
        messages=[
            {"role": "system", "content": "あなたは日本語学者です．"},
            {"role": "user", "content": prompt}
        ]
    )
    result_strength = response.choices[0].message.content.strip()
    return int(result_strength)



# 
# 

def strength_article(text):
    prompt = f"""
    この文章:「 {text}」について，感情極性の強度をそれぞれ-3～3で出力してください．
    強度の種類は「-3:強いネガティブ，-2:ネガティブ，-1:ややネガティブ，0:ニュートラル，1:ややポジティブ，2:ポジティブ，3:強いポジティブ」です．
    出力は「-3,-2, -1, 0, 1, 2,3」の数値のうちいずれか一つを出力してください．
    """
    response = openai.ChatCompletion.create(
        model=FT_model,
        messages=[
            {"role": "system", "content": "あなたは日本語学者です．"},
            {"role": "user", "content": prompt}
        ]
    )
    result_artstrength = response.choices[0].message.content.strip()
    return int(result_artstrength)


# 
# 

def attribute(comment,article):
    prompt = f"""
    以下の文章を次のカテゴリの1つまたは複数に分類してください．出力するのは「意見・根拠・解決策・経験談・非建設」のカテゴリのみです：
    1.意見：自分の意見をもとに議論を起こそうとしている
    2.根拠：客観的で根拠が提示されている
    3.解決策：新たな考え方や解決策を提供している
    4.経験談：記事に関する珍しい経験談である
    5.非建設：当てはまらない
    
    文章: 「{comment}」
    
    出力形式：「意見・根拠・解決策・経験談・非建設」のいずれかのみをカンマで区切って入力
    
    以下に各属性の例文と，その属性に当てはまる根拠を提示します．例文と根拠を参考に段階的に考えて文章を分類してください．
    
    ##例文
    1. 自分の意見をもとに議論を起こそうとしている  
       - 「私は，リモートワークが普及しても対面でのコミュニケーションの重要性は変わらないと考えています．」  
       → この部分で，筆者は自分の意見を述べています．「皆さんはどう思いますか？」と問いかけていることで，他の読者や聞き手に対して議論を喚起し，反応を促そうとしている．個人の考えに基づく意見の表明が議論の起点となっている．
    
    2. 客観的で根拠が提示されている  
       - 「最新の調査によれば，ハイブリッドワークを導入している企業の生産性は平均で15％向上していることが分かっています．」  
       → ここでは，具体的な調査結果という客観的なデータを提示している．数値として生産性の向上や従業員満足度の上昇といった明確な根拠を示しており，その結果をもとにリモートワークとオフィスワークの利点が論じられている．このように，事実やデータに基づいた主張が「客観的で根拠が提示されている」状態に該当する．
    
    3. 新たな考え方や解決策を提供している  
       - 「月に一度の『オフィスデー』を設定してみてはどうでしょうか？」  
       → この文では，リモートワークと対面コミュニケーションの融合という問題に対して，新しい提案である「オフィスデー」の導入を提案している．これは，現状に対する具体的な解決策を提示しているため，「新たな考え方や解決策を提供している」文章となる．
    
    4. 記事に関する珍しい経験談である  
       - 「リモートワーク初期の頃，オフィスに行く必要がある日に，つい自宅からズボンを履かずにビデオ会議に出席してしまったことがありました．」  
       → ここでは，筆者がリモートワークに関連した個人的な体験談を述べている．この経験は特に珍しいものであり，他の人があまり語らないユニークなエピソードである．そのため，このエピソードは「珍しい経験談」に該当する．

    5. 当てはまらない
       -  記事本文の内容{article}と関連しておらず，誰かを誹謗中傷する内容，性的表現や差別的な表現がされていたらこの項目に該当する．

    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "あなたは日本語学者です．"},
            {"role": "user", "content": prompt}
        ]
    )
    result = response.choices[0].message.content.strip()
    return result


# 
# 

def stance(article, comment):
    prompt = f"""
    この記事：「{article}」に寄せられたコメント:「 {comment}」のFAVOR・AGAINSTを判定してください．
    判定の種類は「FAVOR，AGAINST」です．
    出力は「FAVOR，AGAINST」のうちいずれか一つを出力してください．
    """
    response = openai.ChatCompletion.create(
        model=FT_model,
        messages=[
            {"role": "system", "content": "あなたは日本語学者です．"},
            {"role": "user", "content": prompt}
        ]
    )
    result = response.choices[0].message.content.strip()
    return result

# 
# 


nlp = spacy.load("ja_ginza")
def cosine_similarity(comment, article):
    doc1, doc2 = nlp(comment), nlp(article)
    return doc1.similarity(doc2)



# ↓↓↓↓テスト↓↓↓↓
# comment = "やはり来月で3年目に差し掛かるこの侵攻を僅か24時間で終わらす事は不可能でしょうね。それは仕方が無いとしてこの次期大統 領が出来れば今年中にウクライナが一方的に不利な条件に立たされる中途半端な停戦ではなく、ロシアと言う侵略国家が2度とウクラ イナのみならず世界に牙を剥く事がない状態に立たされるような完全な終戦を期待しております"
# article = "トランプ次期米大統領は7日の記者会見で、自らが実現を目指すロシアとウクライナの停戦が容易ではないとの認識をにじませた。これまでは「大統領就任前」や「就任後24時間以内」の停戦実現に意欲を示してきたが、今回は「（停戦まで）6カ月あれば良い」などと説明。プーチン露大統領との会談実現も、20日の就任以降になるとの見方を示し、目標を事実上後退させた格好だ。トランプ氏は会見で、停戦の実現について「6カ月あれば良い。それよりずっと前に解決できることを望む」と説明。プーチン氏との会談については、「プーチン氏は会いたいと思っているだろうが、20日以降でないと適切ではない」とした。その上で、「毎日多くの若者が殺されている」と述べ、早期停戦の必要性を改めて訴えた。トランプ氏はまた、ウクライナが求める北大西洋条約機構（NATO）への加盟に否定的な立場も改めて示した。「ロシアはプーチン氏が就任するずっと前から、NATOがウクライナに関わることはできないと言い続けてきた」と指摘。その上で、バイデン米大統領がウクライナの加盟の可能性に言及したとし、「ロシアの感情は理解できる」と主張した。一方、ロイター通信によると、トランプ次期政権のウクライナ・ロシア担当特使のキース・ケロッグ氏が、1月初旬に予定していたウクライナなどへの訪問をトランプ氏が就任する20日以降に延期したと報じた。停戦を巡っては、敵対する双方の主張に隔たりが大きく、仮にトランプ次期政権の仲介で停戦交渉が始まっても難航が予想されている。"
# attribute = attribute_with_gpt(comment,article)
# print(attribute)
# stance = stance_with_gpt(article,comment)
# print(stance)    

