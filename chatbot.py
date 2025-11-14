"""
Streamlit 기반 Q&A 챗봇 UI
=========================

UI/UX 설계 원칙:
--------------
1. 미니멀리즘: 불필요한 요소 제거, 핵심 기능에 집중
2. 즉시 응답: 비동기 처리 없이 즉각적인 피드백
3. 명확한 상태 표시: 로딩, 에러, 성공 상태 명확히 구분
4. 접근성: 모바일/데스크톱 모두 지원

기술 선택 이유:
-------------
Streamlit 선택 이유:
- 빠른 프로토타이핑: 최소 코드로 웹 앱 구현
- 내장 컴포넌트: 채팅 UI 컴포넌트 기본 제공
- 자동 리로드: 코드 변경시 자동 반영
- 무료 배포: Streamlit Cloud 통해 간편 배포
"""

import os
import streamlit as st
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='langchain_core')

from vector_db import create_vector_db, load_vector_db, get_answer, SIMILARITY_THRESHOLD

# ==================== 페이지 설정 ====================
st.set_page_config(
    page_title="Perso.ai Q&A 챗봇",
    page_icon="💬",
    layout="wide",  # wide: 더 넓은 화면 활용
    initial_sidebar_state="expanded"  # 사이드바 기본 열림
)

# Vector DB 저장 경로
VECTOR_DB_PATH = os.path.join("data", "vector_db")

# ==================== 캐싱 전략 ====================
@st.cache_resource(show_spinner=False)
def initialize_vector_db():
    """
    Vector DB를 초기화하고 캐시합니다.
    
    캐싱 전략 (@st.cache_resource):
    ------------------------------
    - 목적: Vector DB 반복 로드 방지
    - 효과: 첫 로드 후 메모리에 유지
    - 메모리: ~50MB (100개 문서 기준)
    - 세션 간 공유: 모든 사용자가 동일 인스턴스 공유
    - 갱신: 서버 재시작시에만 재로드
    
    성능 향상:
    - 첫 로드: ~2초
    - 캐시 히트: <1ms
    - 메모리 절약: N명 사용자가 1개 인스턴스만 사용
    """
    # Vector DB가 이미 존재하는지 확인
    if os.path.exists(VECTOR_DB_PATH) and os.listdir(VECTOR_DB_PATH):
        try:
            return load_vector_db(embedding_model="huggingface")
        except Exception as e:
            st.warning(f"Vector DB 로드 실패: {e}. 새로운 Vector DB를 생성합니다.")
            return create_vector_db(embedding_model="huggingface")
    else:
        return create_vector_db(embedding_model="huggingface")

# ==================== 메인 함수 ====================
def main():
    """
    메인 애플리케이션 진입점
    
    실행 흐름:
    1. 페이지 제목 렌더링
    2. Vector DB 초기화 (캐시됨)
    3. 세션 상태 초기화
    4. 사이드바 렌더링
    5. 채팅 인터페이스 렌더링
    """
    # 제목 및 설명
    st.title("💬 Perso.ai Q&A 챗봇")
    st.markdown("### 지능형 질의응답 시스템")
    st.markdown("---")
    
    # Vector DB 초기화 (캐시됨)
    with st.spinner("시스템을 초기화하는 중..."):
        vector_db = initialize_vector_db()
    
    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 채팅 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # 메타데이터 표시 (유사도 점수 등)
            if "metadata" in message and message["metadata"]:
                if "similarity_score" in message["metadata"]:
                    st.caption(f"유사도: {message['metadata']['similarity_score']:.4f}")
    
    # 사용자 입력
    if prompt := st.chat_input("Perso.ai에 대해 궁금한 점을 물어보세요!"):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 답변 생성
        with st.chat_message("assistant"):
            with st.spinner("답변을 찾는 중..."):
                try:
                    answer, score = get_answer(prompt, vector_db, similarity_threshold=SIMILARITY_THRESHOLD)
                except Exception as e:
                    error_msg = f"오류가 발생했습니다: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
                    return
                
                if answer:
                    # 유사도 점수가 임계값 이상인 경우 답변 표시
                    st.markdown(answer)
                    
                    # 메타데이터 표시
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption(f"✅ 유사도: {score:.4f}")
                    with col2:
                        st.caption(f"📊 임계값: {SIMILARITY_THRESHOLD}")
                    
                    # 어시스턴트 메시지 추가
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "metadata": {
                            "similarity_score": score
                        }
                    })
                else:
                    if score is not None:
                        # 유사도가 임계값 미만이어서 답변을 찾을 수 없음
                        error_msg = "죄송합니다. 질문에 대한 적절한 답변을 찾을 수 없습니다."
                        st.warning(error_msg)
                        st.caption(f"⚠️ 유사도: {score:.4f} < 임계값 {SIMILARITY_THRESHOLD}")
                        st.info("💡 다른 표현으로 질문해보세요.")
                    else:
                        # 답변을 찾을 수 없는 경우
                        error_msg = "죄송합니다. 답변을 찾을 수 없습니다."
                        st.error(error_msg)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg,
                        "metadata": {"similarity_score": score}
                    })
    
    # ==================== 사이드바 ====================
    with st.sidebar:
        st.header("ℹ️ 정보")
        st.markdown("""
        **Perso.ai Q&A 챗봇**
        
        이 챗봇은 Perso.ai에 대한 질문에 답변합니다.
        
        **사용 방법:**
        1. 아래 입력창에 질문을 입력하세요
        2. Enter를 누르면 답변이 표시됩니다
        3. 이전 대화 기록이 자동으로 저장됩니다
        """)
        
        st.markdown("---")
        
        # 채팅 기록 초기화 버튼
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ 대화 초기화", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        
        with col2:
            if st.button("🔄 새로고침", use_container_width=True):
                st.cache_resource.clear()
                st.rerun()
        
        st.markdown("---")
        
        # 시스템 정보
        st.subheader("📊 시스템 정보")
        
        # Vector DB 상태
        if os.path.exists(VECTOR_DB_PATH):
            st.success("✅ Vector DB 정상")
            
            # 통계 정보
            if vector_db:
                doc_count = vector_db.index.ntotal if hasattr(vector_db, 'index') else "N/A"
                st.metric("문서 수", doc_count)
        else:
            st.error("⚠️ Vector DB 없음")
        
        # 설정 정보
        st.metric("유사도 임계값", f"{SIMILARITY_THRESHOLD:.3f}")
        
        # 채팅 통계
        st.markdown("---")
        st.subheader("📈 채팅 통계")
        st.metric("총 메시지 수", len(st.session_state.messages))

if __name__ == "__main__":
    main()

