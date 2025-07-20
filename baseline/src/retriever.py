import docx2txt
from rank_bm25 import BM25Okapi
from konlpy.tag import Mecab
from tqdm import tqdm
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDING_MODEL_NAME = 'jhgan/ko-sbert-nli'
MECAB_DIC_PATH = 'C:/workspace/malpyeong/venv/Lib/site-packages/mecab-ko-dic'

class Retriever:
    def __init__(self, docx_path):
        self.corpus = self.knowledge_base(docx_path)
        try:
            self.mecab = Mecab(dicpath=MECAB_DIC_PATH)
        except Exception as e:
            raise SystemExit("Mecab 로딩 실패")

        # Sparse Retriever (BM25) 와 Dense Retriever (SBERT) 초기화
        self.bm25, self.sbert_model, self.corpus_embeddings = self._initialize_retrievers()

    def knowledge_base(self, docx_path):
        if not os.path.exists(docx_path):
            raise FileNotFoundError(f"지식 베이스 파일을 찾을 수 없습니다: {docx_path}")
            
        full_text = docx2txt.process(docx_path)
        
        knowledge_corpus = []
        current_chunk = ""
        for line in full_text.splitlines():
            line = line.strip()
            if not line: continue
            if line.startswith('<') and line.endswith('>'):
                if current_chunk:
                    knowledge_corpus.append(current_chunk.strip())
                current_chunk = line + "\n"
            else:
                current_chunk += line + "\n"
        if current_chunk:
            knowledge_corpus.append(current_chunk.strip())
        print(f"   - 총 {len(knowledge_corpus)}개의 청크을 생성했습니다.")

        return knowledge_corpus

    def _initialize_retrievers(self):

        tokenized_corpus = [self.mecab.morphs(doc) for doc in tqdm(self.corpus, desc="   - 형태소 분석")]
        bm25 = BM25Okapi(tokenized_corpus)

        # SBERT (Dense) 초기화
        sbert_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        corpus_embeddings = sbert_model.encode(self.corpus, convert_to_tensor=True, show_progress_bar=True)
        
        return bm25, sbert_model, corpus_embeddings

    def search(self, query, top_n=1):
        # 1단계: BM25로 1차 후보군 검색 (top_n의 5배수만큼 넉넉하게)
        tokenized_query = self.mecab.morphs(query)
        
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # 점수가 높은 순으로 정렬하되, 후보군을 top_n보다 많이 가져옴
        candidate_indices = np.argsort(bm25_scores)[::-1][:top_n * 5].copy()
        
        # 2단계: SBERT와 코사인 유사도로 재정렬 (Re-ranking)
        query_embedding = self.sbert_model.encode(query, convert_to_tensor=True)
        candidate_embeddings = self.corpus_embeddings[candidate_indices]
        
        cosine_scores = cosine_similarity(
            query_embedding.cpu().numpy().reshape(1, -1),
            candidate_embeddings.cpu().numpy()
        )[0]
        
        reranked_indices_local = np.argsort(cosine_scores)[::-1]
        
        # 최종 top_n개의 문서 반환
        final_indices = [candidate_indices[i] for i in reranked_indices_local]
        retrieved_docs = [self.corpus[i] for i in final_indices[:top_n]]
        
        return retrieved_docs