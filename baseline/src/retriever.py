# 이 파일은 'src/retriever.py'로 저장되어야 합니다.

import hashlib
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from konlpy.tag import Mecab,Okt
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# --- 상수 정의 ---
EMBEDDING_MODEL_NAME = 'jhgan/ko-sbert-nli'
RRF_K = 60  # Reciprocal Rank Fusion 상수

class Retriever:
    def __init__(self, knowledge_base_path: str, cache_dir: str = "./cache"):
        """
        Retriever를 초기화
        """
        print("Retriever 초기화 시작...")
        self.json_path = Path(knowledge_base_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.tokenizer = Mecab()
            self.tokenizer_name = "mecab"
            print("Mecab 형태소 분석기를 사용합니다.")
        except Exception:
            print(f"경고: Mecab 로딩 실패. Okt 형태소 분석기로 대체합니다.")
            print("성능 향상을 위해 Mecab 설치를 권장합니다: https://konlpy.org/en/latest/install/#install-mecab")
            self.tokenizer = Okt()
            self.tokenizer_name = "okt"

        self.sbert_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        if not self._is_cache_valid():
            print("캐시가 유효하지 않거나 없습니다. 캐시를 새로 생성합니다.")
            self._clear_cache()
            self.knowledge_base = self._load_knowledge_base_from_file()
            self.corpus = [doc['content'] for doc in self.knowledge_base]
            self.bm25, self.corpus_embeddings = self._create_and_cache_retrievers()
        else:
            print("유효한 캐시를 발견했습니다. 캐시에서 데이터를 로드합니다.")
            self.knowledge_base, self.corpus, self.bm25, self.corpus_embeddings = self._load_from_cache()

        print("Retriever 초기화 완료.")

    def _get_file_hash(self) -> str:
        """JSON 파일의 SHA256 해시를 계산하여 반환합니다."""
        sha256_hash = hashlib.sha256()
        with open(self.json_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _is_cache_valid(self) -> bool:
        """캐시가 최신 상태인지 확인합니다."""
        info_path = self.cache_dir / "cache_info.json"
        if not info_path.exists(): return False

        with open(info_path, "r", encoding="utf-8") as f:
            cache_info = json.load(f)
        
        if cache_info.get("file_hash") != self._get_file_hash():
            print("원본 JSON 파일이 변경되었습니다.")
            return False
        
        required_files = ["knowledge_base.pkl", "tokenized_corpus.pkl", "corpus_embeddings.pt"]
        return all((self.cache_dir / f).exists() for f in required_files)

    def _clear_cache(self):
        """캐시 디렉토리의 모든 파일을 삭제합니다."""
        print("기존 캐시를 삭제합니다...")
        for file in self.cache_dir.glob('*'):
            file.unlink()

    def _load_knowledge_base_from_file(self) -> List[Dict[str, Any]]:
        """지정된 경로에서 JSON 파일을 로드합니다."""
        if not self.json_path.exists():
            raise FileNotFoundError(f"지식 베이스 파일을 찾을 수 없습니다: {self.json_path}")
        
        print(f"'{self.json_path}'에서 지식 베이스 로딩 중...")
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  - 총 {len(data)}개의 문서 로드 완료.")
        return data

    def _create_and_cache_retrievers(self) -> Tuple[BM25Okapi, torch.Tensor]:
        """BM25와 SBERT 임베딩을 생성하고 캐시 파일로 저장합니다."""
        print("코퍼스 토큰화 및 임베딩 생성 중...")
        
        tokenized_corpus = [self.tokenizer.morphs(doc) for doc in tqdm(self.corpus, desc="  - 형태소 분석")]
        bm25 = BM25Okapi(tokenized_corpus)

        corpus_embeddings = self.sbert_model.encode(
            self.corpus, convert_to_tensor=True, show_progress_bar=True
        )

        print("생성된 데이터를 캐시에 저장합니다...")
        with open(self.cache_dir / "knowledge_base.pkl", "wb") as f: pickle.dump(self.knowledge_base, f)
        with open(self.cache_dir / "tokenized_corpus.pkl", "wb") as f: pickle.dump(tokenized_corpus, f)
        torch.save(corpus_embeddings, self.cache_dir / "corpus_embeddings.pt")
        
        cache_info = {"file_hash": self._get_file_hash(), "tokenizer": "okt"}
        with open(self.cache_dir / "cache_info.json", "w", encoding="utf-8") as f:
            json.dump(cache_info, f)
            
        return bm25, corpus_embeddings

    def _load_from_cache(self) -> Tuple[List[Dict[str, Any]], List[str], BM25Okapi, torch.Tensor]:
        """캐시 파일에서 데이터를 로드합니다."""
        with open(self.cache_dir / "knowledge_base.pkl", "rb") as f: knowledge_base = pickle.load(f)
        with open(self.cache_dir / "tokenized_corpus.pkl", "rb") as f: tokenized_corpus = pickle.load(f)
        corpus_embeddings = torch.load(self.cache_dir / "corpus_embeddings.pt")
        
        corpus = [doc['content'] for doc in knowledge_base]
        bm25 = BM25Okapi(tokenized_corpus)
        return knowledge_base, corpus, bm25, corpus_embeddings

    def search(self, query: str, top_n: int = 3) -> List[str]:
        """
        RRF를 사용하여 하이브리드 검색을 수행하고, content 문자열 리스트를 반환합니다.
        """
        bm25_top_k = top_n * 5

        # 1. BM25 (Sparse) 검색
        tokenized_query = self.tokenizer.morphs(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_ranked_indices = np.argsort(bm25_scores)[::-1][:bm25_top_k].copy()

        # 2. SBERT (Dense) 검색
        query_embedding = self.sbert_model.encode(query, convert_to_tensor=True).cpu()
        
        if len(bm25_ranked_indices) == 0: return []
        
        candidate_embeddings = self.corpus_embeddings[bm25_ranked_indices].cpu()
        cosine_scores = cosine_similarity(query_embedding.reshape(1, -1), candidate_embeddings)[0]
        sbert_ranked_local_indices = np.argsort(cosine_scores)[::-1]
        sbert_ranked_indices = [bm25_ranked_indices[i] for i in sbert_ranked_local_indices]

        # 3. RRF로 결과 융합
        rrf_scores = {}
        for rank, doc_idx in enumerate(bm25_ranked_indices):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1 / (RRF_K + rank + 1)
        for rank, doc_idx in enumerate(sbert_ranked_indices):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1 / (RRF_K + rank + 1)
            
        sorted_docs_by_rrf = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        final_indices = [doc_id for doc_id, score in sorted_docs_by_rrf]
        
        return [self.corpus[i] for i in final_indices[:top_n]]
