import os
import json
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='langchain_core')

import pandas as pd
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader

class ExcelLoader(BaseLoader):
    """
    Excel 파일에서 Q&A 데이터를 로드하는 로더 클래스.
    
    Excel 파일 형식:
    - 순번, 질문(Q.), 답변(A.) 형식의 Q&A 쌍 데이터
    - 헤더 행은 '순번' 또는 '내용' 키워드로 시작
    
    Returns:
        documents: LangChain Document 객체 리스트 (질문은 page_content, 답변은 metadata)
        qa_list: Q&A 데이터 딕셔너리 리스트
    """
    def __init__(self, file_path: str):
        """
        Args:
            file_path: Excel 파일 경로
        """
        self.file_path = file_path
    
    def load(self):
        """
        Excel 파일을 로드하고 Q&A 쌍을 Document 객체로 변환합니다.
        
        처리 방식:
        - 질문(Q)은 page_content로 저장 → 벡터화되어 검색 대상이 됨
        - 답변(A)은 metadata['answer']로 저장 → 검색 후 최종 반환될 정답
        
        Returns:
            tuple: (documents, qa_list)
                - documents: Document 객체 리스트
                - qa_list: Q&A 데이터 딕셔너리 리스트
        """
        df = pd.read_excel(self.file_path)
        documents = []
        
        # 헤더 행 찾기 (순번, 내용이 있는 행)
        start_idx = None
        for idx, row in df.iterrows():
            row_values = [str(val) for val in row.values if pd.notna(val)]
            if '순번' in ' '.join(row_values) or '내용' in ' '.join(row_values):
                start_idx = idx + 1
                break
        
        if start_idx is None:
            start_idx = 0
        
        # Q&A 데이터 수집
        qa_list = []
        current_qa = {}
        
        for idx in range(start_idx, len(df)):
            row = df.iloc[idx]
            
            # 컬럼에서 값 추출 (Unnamed 컬럼들)
            values = []
            for col in df.columns:
                value = row[col]
                if pd.notna(value) and str(value).strip():
                    val_str = str(value).strip()
                    # 불필요한 컬럼 제외
                    if '[ESTSoft]' not in val_str:
                        values.append(val_str)
            
            if not values:
                continue
            
            # 순번이 있으면 새로운 Q&A 시작
            if len(values) > 0 and values[0].isdigit():
                # 이전 Q&A 저장
                if current_qa and current_qa.get('question') and current_qa.get('answer'):
                    qa_list.append({
                        'id': int(current_qa['number']),
                        'question': current_qa['question'],
                        'answer': current_qa['answer']
                    })
                
                # 새로운 Q&A 시작
                current_qa = {
                    'number': values[0],
                    'question': values[1] if len(values) > 1 else '',
                    'answer': ''
                }
            elif len(values) > 0 and values[0].startswith('Q.'):
                current_qa['question'] = values[0]
            elif len(values) > 0 and values[0].startswith('A.'):
                current_qa['answer'] = values[0]
        
        # 마지막 Q&A 저장
        if current_qa and current_qa.get('question') and current_qa.get('answer'):
            qa_list.append({
                'id': int(current_qa['number']),
                'question': current_qa['question'],
                'answer': current_qa['answer']
            })
        
        # Document 객체로 변환
        # 질문(Q)은 page_content로 저장 (벡터화 대상, 검색 대상)
        # 답변(A)은 metadata로 저장 (최종 반환될 정답)
        for qa in qa_list:
            documents.append(Document(
                page_content=qa['question'],  # 질문: 벡터화되어 검색 대상이 됨
                metadata={
                    'id': qa['id'],
                    'answer': qa['answer']  # 답변: 메타데이터로 저장되어 검색 후 반환됨
                }
            ))
        
        return documents, qa_list

if __name__ == "__main__":
    # Excel 파일에서 데이터 로드
    xlsx_path = os.path.join("data", "qa_data.xlsx")
    xlsx_loader = ExcelLoader(file_path=xlsx_path)
    documents, qa_list = xlsx_loader.load()
    
    # JSON 형태로 변환
    qa_data = {
        'total': len(qa_list),
        'data': qa_list
    }
    
    # JSON 파일로 저장 (data 폴더 안에)
    output_path = os.path.join("data", "qa_data.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(qa_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ JSON 파일이 저장되었습니다: {output_path}")
    print(f"✅ 총 {qa_data['total']}개의 Q&A 데이터가 저장되었습니다.\n")
    
    print("=" * 80)
    print(f"✅ 총 {len(documents)}개의 Q&A 문서가 준비되었습니다.\n")
    
    # 문서 구조 확인 (처음 3개만 출력)
    for i, doc in enumerate(documents[:3], 1):
        print(f"[문서 {i}]")
        print(f"ID: {doc.metadata['id']}")
        print(f"질문 (벡터화 대상): {doc.page_content}")
        print(f"답변 (메타데이터): {doc.metadata['answer']}")
        print("-" * 80)
    
    print(f"\n✅ Q&A 쌍이 하나의 검색 가능한 단위(Document)로 처리되었습니다.")
    print(f"   - 질문(Q): Vector DB에 벡터화되어 저장 (유사도 검색의 대상)")
    print(f"   - 답변(A): 메타데이터로 저장 (검색 후 최종 반환될 정답)")