#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
from konlpy.tag import Okt
from math import log, sqrt
import os
from tqdm import tqdm
import sys
import json
import csv

okt = Okt()

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
RAW_DATA_DIR = PROJECT_DIR + '/dataraw'
FILENAME = 'textrank_extractive.csv'
SUMMARY_DATA = "extractive_test_v2.jsonl"

def PowerMethod(M, epsilon = 1e-06):
    shape = M.shape[0]
    p = np.full((shape, 1), 1/shape)
    prev_p = p.copy()
    while True:
        prev_p = p.copy()
        p = np.dot(M.T, p)
        length = np.linalg.norm(prev_p - p)
        if length < epsilon:
            break
    return p

def convert_to_Markov_chain(M, d = 0.85):
    shape = M.shape[0]
    U = np.full((shape, shape), 1 / shape)
    M = (1 - d) * U + d * M
    return M

def normalize_columns(M, epsilon=1e-6):
    M_sum = M.sum(axis = 1, keepdims = True)
    M_sum = M_sum + epsilon
    M = M / M_sum
    return M

def tokenize_sentences(sentences):
    tokens = []
    for i in range(len(sentences)):
        tokens.append(okt.morphs(sentences[i], norm=True, stem=True))
    return tokens

def count_totalword(tokens):
    words_dict = {}
    for token in tokens:
        for t in token:
            if t in words_dict:
                words_dict[t] += 1
            else:
                words_dict[t] = 1
    return words_dict

def compute_tfidf(words_dict, tokens):
    size = len(tokens)
    tot_tfidf = []
    for token in tokens:
        tfidf = {}
        for t in token:
            if t in tfidf:
                tfidf[t] += log(size / words_dict[t], 2)
            else:
                tfidf[t] = log(size / words_dict[t], 2)
        tot_tfidf.append(tfidf)
    return tot_tfidf

def adjacency_from_tfidf(tfidf, type="lexrank", threshold=0.1):
    size = len(tfidf)
    adj = np.zeros((size, size))
    norm = []
    for i in range(size):
        n = 1e-6
        for key in tfidf[i]:
            n += tfidf[i][key] * tfidf[i][key]
        norm.append(sqrt(n))
    
    for s1 in range(size):
        for s2 in range(size):
            if s1 == s2:
                adj[s1][s2] = 1
            else:
                s1_tfidf = tfidf[s1]
                s2_tfidf = tfidf[s2]
                for k1 in s1_tfidf:
                    for k2 in s2_tfidf:
                        if k1 == k2:
                            adj[s1][s2] += s1_tfidf[k1] * s2_tfidf[k2]
                adj[s1][s2] /= norm[s1]*norm[s2]
                if type == "lexrank":
                    if adj[s1][s2] > threshold:
                        adj[s1][s2] = 1
                    else:
                        adj[s1][s2] = 0
    #np.set_printoptions(precision=3)
    #print(adj)
    return adj

def get_top_three(ans, sentences):
    sorted_index_array = np.argsort(ans, axis = 0)
    return np.sort(sorted_index_array[-3:].reshape((3,)))

#tokens: (n, m) w/ n= # of sentences, m = # of 형태소 in sentences[n]
def adjacency_from_sim(tokens, sentences):
    size = len(tokens)
    adj = np.zeros((size, size))

    for s1 in range(size):
        for s2 in range(size):
            if s1 == s2:
                continue
            for t1 in tokens[s1]:
                for t2 in tokens[s2]:
                    if t1 == t2:
                        adj[s1][s2] += 1
            adj[s1][s2] /= log(len(tokens[s1]) + 1e-6, 2) * log(len(tokens[s2]) + 1e-6, 2)

    #np.set_printoptions(linewidth=np.inf)
    #print(adj)
    return adj

def PowerMethod_textrank(adj, d = 0.85, epsilon=1e-6):
    size = adj.shape[0]
    adj = normalize_columns(adj)
    adj = d * adj
    bias = np.ones((size, 1)) * (1-d)
    p = np.ones((size, 1))
    while True:
        prev_p = p.copy()
        p = np.dot(adj.T, p) + bias
        length = np.linalg.norm(prev_p - p)
        if length < epsilon:
            break
    return p

def sentence_to_topthree(sentences, type="lexrank", threshold = 0.1):

    tokens = tokenize_sentences(sentences)
    if type=="lexrank" or type=="continuous_lexrank":
        words_dict = count_totalword(tokens)
        tfidf = compute_tfidf(words_dict, tokens)
        adj = adjacency_from_tfidf(tfidf, type, threshold)
        adj = normalize_columns(adj)
        if type == "lexrank":
            adj = convert_to_Markov_chain(adj)
        ans = PowerMethod(adj)
    elif type=="textrank":
        adj = adjacency_from_sim(tokens, sentences)
        ans = PowerMethod_textrank(adj)

    top_three = get_top_three(ans, sentences)
    return top_three

def top_three_sentences(sentences, indexes):
    ans = ""
    ans += sentences[indexes[0]] + "\n"
    ans += sentences[indexes[1]] + "\n"
    ans += sentences[indexes[2]]
    print("summarized texts:")
    print(indexes[0], sentences[indexes[0]])
    print(indexes[1], sentences[indexes[1]])
    print(indexes[2], sentences[indexes[2]])
    return ans

def save_to_files(summaries):
    file = open(PROJECT_DIR + FILENAME, mode="w", encoding="UTF-8", newline="")
    writer = csv.writer(file)
    writer.writerow(["id", "summary"])
    for summary in summaries:
        writer.writerow(list(summary.values()))
    return 

def get_train():
    sys.path.insert(0, PROJECT_DIR)
    with open(RAW_DATA_DIR + '/' + SUMMARY_DATA, 'r', encoding="UTF-8") as json_file:
        json_list = list(json_file)

    trains = []
    for json_str in json_list:
        line = json.loads(json_str)
        trains.append(line)
    return trains

def get_summaries_from_texts(trains):
    summaries = []

    for i in range(len(trains)):
        t = trains[i]
        sentences = t["article_original"]

        top_three = sentence_to_topthree(sentences, "textrank")
        extractive_summary = top_three_sentences(sentences, top_three)
        summaries.append({"id": t["id"], "summary": extractive_summary})
        
        if len(summaries) % 100 == 0:
            print(len(summaries))
    return summaries

trains = [
    {"id": "0000013024", 
    "article_original": [
        "비트코인이 전날에 비해 소폭 상승세를 보이며 5500만원대에 안착했다.", 
        "일론 머스크 테슬라 최고경영자(CEO)는 트위터에서 \"보유 중인 비트코인을 매도하지 않았다\"고 해명했지만 큰 폭의 가격 반등은 일어나지 않는 모양새다.",
        "18일 오후 4시 30분 기준, 암호화폐 거래소 업비트에 따르면 비트코인은 1BTC당 24시간 전 대비 2.15% 오른 5515만원대를 기록 중이다.",
        "비트코인은 지난 14일 종가 기준, 6100만원대를 기록한 이후 꾸준히 하락세를 보이고 있다.", "지난주 머스크는 테슬라 전기차 비트코인 결제 중단을 선언하며 가격 하락을 불러온 바 있다.",
        "특히 머스크는 지난 16일(현지시각) 테슬라 소유 비트코인을 매각할 수 있다고 시사하면서 시세가 요동쳤다.", 
        "이에 지난 17일(현지시각) 트위터에서 비트코인 관련, 시장 분석가로 활동하는 익명 투자자 비트코인 아카이브가 \"일론 머스크가 영양가 없는 글을 써 비트코인이 약 20% 하락했다\"며 \"당신은 왜 사람들이 화가 났는지 알고 있는가?\"라는 내용의 글을 게시했고, 머스크는 \"명확히 얘기하겠다\"며 \"보유중인 비트코인을 매도하지 않았다\"고 답변했다.",
        "머스크의 해명 이후 비트코인 가격은 소폭 상승했지만 큰 폭의 반등은 보여주지 못하고 있다.", "전날 종가 기준 5400만원대였던 비트코인 가격은 현재 5500만원에서 상승세가 멈춘 상황이다.", "코인마켓캡에서도 비트코인 가격은 4만5000달러대까지 오르는 데 그쳤다.",
        "이더리움은 1ETH(이더리움 단위)당 4.67% 오른 428만원에 거래 중이다.", "도지코인은 1DOGE(도지코인 단위)당 전일대비 0.17% 하락한 603원에, 리플은 1XRP(리플 단위)당 전일대비 0.53% 내린 1850원에 거래되고 있다.",
        "이더리움클래식은 1ETC(이더리움클래식 단위)당 전일대비 0.94% 오른 11만2450원에 거래 중이다."
        ]
    }, {
        "id": "0000719385", 
        "article_original": [
            "현대코퍼레이션(011760)(옛 현대종합상사)이 차량용 플라스틱 부품 제작사이자 현대자동차 1차 협력사인 신기인터모빌 인수를 추진한다.", 
            "18일 업계에 따르면 현대코퍼레이션은 이날 오후 매각이 진행 중인 신기인터모빌 인수를 위한 우선협상대상자에 선정됐다.", 
            " 인수 대상은 경영권 포함, 신기인터모빌 지분 70%인 것으로 알려졌다.", 
            "신기인터모빌은 1987년 현대차 협력업체로 등록된 이후 지난 33년간 콘솔박스, 엔진커버, 휠가드, 내장트림 등 고기능 경량화 플라스틱 부품을 현대기아차에 주력 공급해 왔다.", 
            "현대코퍼레이션은 신기인터모빌이 정밀 플라스틱 사출 가공 분야에 경쟁력이 있다고 설명했다.", 
            "이 회사의 플라스틱 제품이 전동화와 영향이 없는 제품들이며 경량화에 대한 수요 증가로 미래 성장 가능성이 높은 기업으로 평가받고 있다고 한다.", 
            "앞서 지난 3월 현대코퍼레이션은 사업 영역 확대·다변화를 위해 새롭게 사명을 변경하고 자동차 및 전기차 부품 제조, 친환경 소재 및 복합 소재와 친환경 에너지 인프라 구축 관련 사업 등을 목적 사업에 추가한 바 있다.", 
            "현대코퍼레이션은 이번 인수 협상이 순조롭게 진행될 경우 경량 플라스틱 제조 및 사출 분야에 우수한 기술력을 보유한 신기인터모빌의 기술력을 바탕으로 제조업 분야 기반을 강화할 방침이다.", 
            "회사의 기존 모빌리티 사업이 확보하고 있는 글로벌 네트워크와의 시너지 창출, 해외 자동차 제조사를 상대로 한 부품 수출 시장 개척 등을 본격화해 나갈 전망이다.", 
            "현대코퍼레이션 관계자는 \"추후 진행되는 본실사와 협상에 성실히 임하면서 인수 대상 회사의 경쟁력 강화와 인수 후 통합 과정을 통한 시너지 등 주요 이슈 사항들을 정밀하게 점검할 예정\"이라고 말했다."
        ]
    }, {
        "id": "0003103375",
        "article_original": [
            "삼성전자와 SK하이닉스 등 국내 반도체 기업이 주도하고 있는 메모리 반도체 시장이 올해 호황기에 진입해 내년 사상 최대 매출을 경신할 것이라는 전망이 나왔다.", 
            "21일 글로벌 시장조사업체 IC인사이츠는 “세계 메모리 반도체 전체 매출액이 내년 1804억 달러(약 204조원)를 기록하며 역대 최대 기록을 갈아치울 것”이라고 예상했다.", 
            "내년 이후에도 호황이 이어져 2023년 매출액 2196억 달러(약 249조원)로 정점을 찍을 것으로 전망했다.", 
            "지금까지 메모리 반도체 시장의 연간 최대 매출액은 2018년에 올린 1633억 달러(약 185조원)이었다.", 
            "IC인사이츠는 올해도 D램 가격의 빠른 상승세에 힘입어 메모리 반도체 전체 매출액 1552억 달러(175조원)를 달성할 것으로 내다봤다.", 
            "지난해(1267억 달러) 대비 23% 증가한 수치다.", 
            "이중 D램 매출이 56%, 낸드플래시가 41%를 차지할 것으로 IC인사이츠는 분석했다.", 
            "이미 삼성전자와 SK하이닉스는 1분기 실적을 발표하는 컨퍼런스콜에서 올해 2분기 이후 업황에 대해 ‘긍정적’으로 전망했다.", 
            "현재 전 세계 D램 시장 점유율은 삼성전자가 42%로 1위, SK하이닉스가 29%로 2위다.", 
            "낸드플래시 역시 삼성전자가 32% 점유율로 1위고, SK하이닉스는 인수를 앞둔 인텔 낸드 사업부와 합쳐서 계산할 경우 점유율 20%대로 2위다."
        ]
    },{
        "id": "0004642969",
        "article_original": [
            "스테판 반셀 모더나 최고경영자(CEO)가 자사의 신종 코로나바이러스 감염증(코로나19) 백신을 일본에서 생산할 것을 검토하고 있다고 21일 밝혔다.", 
            "니혼게이자이신문에 따르면 반셀 CEO는 \"일본을 포함한 아시아에서 백신 생산을 검토하고 있으며 협의를 진행하고 있다\"며 \"아직 초기 단계지만 일본의 여러 전문가와 협의를 진행하고 있다\"고 말했다.", 
            "신문은 모더나가 일본 기업과 위탁생산(CMO) 계약이나 라이선스 계약을 맺을 가능성이 있다고 내다봤다.", 
            "반셀 CEO는 \"아시아에서의 사업 및 생산 확장에 대해 매우 흥미를 갖고 있다\"며 \"일본의 높은 노동력과 연구력에 대해 이해하고 있다\"고 설명했다.", 
            "일본 정부는 모더나와 9월까지 백신 5000만회분을 공급받는 계약을 체결했으며, 이날 모더나 백신을 정식 승인했다.", 
            "아울러 반셀 CEO는 \"도쿄올림픽 전에 승인이 난 것을 환영한다\"며 \"일본의 임상시험에서도 미국의 임상시험과 마찬가지로 충분한 효과가 나타났다\"고 자사 백신의 효과를 자신했다.", 
            "그는 추가 계약 가능성이 있는지에 대한 질문에 \"내년에 대비하여 일본 정부를 포함해 협의를 진행하고 있다\"고 언급했다."
        ]
    },{
        "id": "0000001200",
        "article_original": [
            "넷마블의 북미 지역 자회사인 잼시티(Jam City)가 기업인수목적회사(스팩·SPAC)와의 합병을 통해 미국 증시에 상장을 추진한다.", 
            "아울러 합병을 통해 마련한 자금으로 캐나다 게임 개발사 인수와 신작 개발 등에 나설 예정이다.", 
            "넷마블은 잼시티가 기업인수목적회사인 DPCM과 합병해 미국 뉴욕증권거래소에 상장할 계획이라고 21일 밝혔다.", 
            "이번 합병을 통해 확보하는 4억달러의 자금 가운데 일부는 캐나다 개발사 '루디아(Ludia Inc.)' 인수에 사용한다.", 
            "나머지는 신작 개발과 기술력 강화, 인수합병(M&A) 등을 위해 활용할 예정이다.", 
            "루디아는 '쥬라기 월드: 얼라이브'와 '드래곤즈: 타이탄 업라이징' 등 세계적인 지적재산권(IP)을 기반으로 다양한 장르의 모바일 게임을 개발 및 퍼블리싱하고 있다.", 
            "잼시티의 공동 창립자이자 CEO인 크리스 디울프(Chris DeWolfe)는 \"잼시티는 세계 최고 수준의 모바일 엔터테인먼트 플랫폼을 구축하여 꾸준히 확장해 나가고 있다\"며 \"이번 합병을 통해 확보한 자금으로 성장을 가속화하고 루디아의 훌륭한 개발진과 함께 유저 친화형 게임을 선보이겠다\"고 말했다.", 
            "잼시티는 넷마블이 2015년 7월 게임 개발 및 해외 거점 확보를 위해 1497억원을 들여 지분 60% 가량을 사들인 모바일 개발사다.", 
            "'쿠키잼'과 '해리포터: 호그와트 미스터리', '디즈니 이모지 블리츠' 등을 개발했다.", 
            "TinyCo와 JCSA, JCTO 스튜디오 등 8개 계열사를 거느리고 있다.", 
            "지난해 매출 4453억원에 순이익 309억원을 달성했다.", 
            "지난해에는 앱스토어와 구글플레이의 미국 10대 게임 퍼블리셔로 선정된 바 있다."
        ]
    }
]

#trains = get_train()
#trains = [trains[0]]
summaries = get_summaries_from_texts(trains)
#save_to_files(summaries)