from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import time
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


def csv_loader(file_path):
    loader = CSVLoader(file_path)
    return loader.load()


load_dotenv()


def create_vector_store(_docs, batch_size=150):
    # 벡터 스토어 초기화 (빈 상태로 시작)
    vectorstore = None
    total_docs = len(_docs)
    print(f"총 문서 수: {total_docs}개")

    for i in range(0, total_docs, batch_size):
        # 100개씩 슬라이싱
        batch_docs = _docs[i:i + batch_size]

        # 현재 배치 처리 중임을 출력
        print(f"{i + 1}번째부터 {i + len(batch_docs)}번째 문서까지 처리 중...")

        # 벡터화 및 벡터 스토어 생성 또는 업데이트
        if vectorstore is None:
            # 첫 번째 배치는 새로 벡터 스토어를 생성
            vectorstore = Chroma.from_documents(
                batch_docs,
                OpenAIEmbeddings(model='text-embedding-3-small'),
                persist_directory="./db"
            )
        else:
            # 그 이후의 배치는 기존 벡터 스토어에 추가
            vectorstore.add_documents(batch_docs)

        # 100개 처리 후 1분 대기 (TPM 초과 방지)
        if i + batch_size < total_docs:
            print("1분 대기 중...")
            time.sleep(61)  # 1분 대기

    return vectorstore


csv = csv_loader('./files/refined_data.csv')
create_vector_store(csv)
