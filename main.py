# 라이브러리 및 모듈 가져오기
import streamlit as st  # Streamlit을 사용하여 웹 애플리케이션 생성
from pathlib import Path  # 파일 경로 작업을 위한 Path 클래스
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader  # 다양한 파일 형식에서 텍스트 추출하는 로더들
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 텍스트를 작은 청크로 나누기 위한 모듈  
from langchain_huggingface import HuggingFaceEmbeddings # HuggingFace 모델을 통한 텍스트 임베딩 처리
from langchain_community.vectorstores import FAISS  # FAISS 벡터 스토어를 통해 텍스트 검색 기능 구현
from langchain_community.callbacks import get_openai_callback  # OpenAI 응답을 받아오는 콜백
from langgraph.checkpoint.memory import MemorySaver  # LangGraph의 메모리 관리
from langchain_openai import ChatOpenAI  # OpenAI 언어 모델 사용을 위한 모듈
from langgraph.graph import START, MessagesState, StateGraph  # LangGraph의 StateGraph
from langchain_core.messages import HumanMessage, AIMessage  # 사용자와 AI 메시지를 나타내는 스키마
import tiktoken  # 토큰화 처리를 위한 모듈
import json  # JSON 형식의 데이터 관리
import base64  # 텍스트 인코딩을 위해 Base64 사용
import speech_recognition as sr  # 음성 인식 기능을 위한 모듈
import tempfile  # 임시 파일 생성 및 관리 모듈
import uuid

# LangGraph를 통해 대화 워크플로 설정
workflow = StateGraph(state_schema=MessagesState)

# OpenAI 모델 정의
model = ChatOpenAI()

# 모델 호출 함수 정의
def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}  # 메시지 리스트에 응답을 추가하여 반환

# 노드와 메모리 설정
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# 애플리케이션 실행 함수 정의
def main():
    # 페이지 설정 (Streamlit 상단 바 구성)
    st.set_page_config(page_title="에너지", page_icon="🌻", layout="centered")  # 레이아웃을 설정하여 폰트 경고를 방지
    st.image('energy.png')  # 상단에 이미지를 표시
    st.title("_:red[에너지 학습 도우미]_ 🏫")  # 제목 표시 (에너지 학습 도우미)
    st.header("😶주의! 이 챗봇은 참고용으로 사용하세요!", divider='rainbow')  # 주의사항 표시

    # 세션 상태 초기화
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())  # 고유 스레드 ID 생성
    if "voice_input" not in st.session_state:  # voice_input 속성 초기화
        st.session_state.voice_input = ""  # 빈 문자열로 초기화

    # 사이드바 구성
    with st.sidebar:
        folder_path = Path()  # 텍스트 파일이 있는 폴더 경로 (현재 경로)
        openai_api_key = st.secrets["OPENAI_API_KEY"]  # OpenAI API 키 설정 (Streamlit secrets 사용)
        model_name = 'gpt-4o-mini'  # 사용할 OpenAI 모델 이름 설정
        
        # 사이드바에 안내 메시지 및 Process 버튼
        st.text("아래의 'Process'를 누르고\n아래 채팅창이 활성화 될 때까지\n잠시 기다리세요!😊😊😊")
        process = st.button("Process", key="process_button")  # Process 버튼 추가하여 모델 초기화 및 준비

        if process:
            files_text = get_text_from_folder(folder_path)  # 폴더에서 텍스트 추출
            text_chunks = get_text_chunks(files_text)  # 텍스트를 분할하여 청크로 변환
            vectorstore = get_vectorstore(text_chunks)  # 텍스트 임베딩 및 벡터 스토어 생성
            st.session_state.processComplete = True  # 처리 완료 상태로 설정

        # 음성 입력을 받아 녹음하고 텍스트로 변환
        audio_value = st.experimental_audio_input("음성 메시지를 녹음하여 질문하세요😁.")
        
        if audio_value:
            with st.spinner("음성을 인식하는 중..."):
                recognizer = sr.Recognizer()
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                        temp_audio_file.write(audio_value.getvalue())
                        with sr.AudioFile(temp_audio_file.name) as source:
                            audio = recognizer.record(source)
                            st.session_state.voice_input = recognizer.recognize_google(audio, language='ko-KR')
                    st.session_state.voice_input = st.session_state.voice_input.strip()
                except sr.UnknownValueError:
                    st.warning("음성을 인식하지 못했거나 모델을 불러오지 않았습니다. Process를 눌르고 다시 시도하세요!")
                except sr.RequestError:
                    st.warning("서버와의 연결에 문제가 있습니다. 다시 시도하세요!")
                except OSError:
                    st.error("오디오 파일을 처리하는 데 문제가 발생했습니다. 다시 시도하세요!")

        save_button = st.button("대화 저장", key="save_button")
        if save_button:
            if st.session_state.chat_history:
                save_conversation_as_txt(st.session_state.chat_history)
            else:
                st.warning("질문을 입력받고 응답을 확인하세요!")
                
        clear_button = st.button("대화 내용 삭제", key="clear_button")
        if clear_button:
            st.session_state.chat_history = []
            st.session_state.thread_id = str(uuid.uuid4())

    # 질문 입력 필드
    query = st.session_state.voice_input or st.chat_input("질문을 입력해주세요.")

    # 질문이 있을 경우 대화 체인에서 응답 생성
    if query:
        st.session_state.voice_input = ""
        try:
            input_message = HumanMessage(content=query)
            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
                st.session_state.chat_history.append((input_message.content, event["messages"][-1].content))

            # 대화 내역 출력
            for user_msg, ai_msg in st.session_state.chat_history:
                st.write(f"**User:** {user_msg}")
                st.write(f"**AI:** {ai_msg}")
        except Exception as e:
            st.error("질문을 처리하는 중 오류가 발생했습니다. 다시 시도하세요.")

# 텍스트 토큰 길이 계산 함수
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

# 폴더에서 텍스트 추출
def get_text_from_folder(folder_path):
    doc_list = []
    folder = Path(folder_path)
    files = folder.iterdir()

    for file in files:
        if file.is_file():
            if file.suffix == '.pdf':
                loader = PyPDFLoader(str(file))
                documents = loader.load_and_split()
            elif file.suffix == '.docx':
                loader = Docx2txtLoader(str(file))
                documents = loader.load_and_split()
            elif file.suffix == '.pptx':
                loader = UnstructuredPowerPointLoader(str(file))
                documents = loader.load_and_split()
            else:
                documents = []
            doc_list.extend(documents)
    return doc_list

# 텍스트 분할 함수
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    return text_splitter.split_documents(text)

# 벡터 저장소 생성 함수
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return FAISS.from_documents(text_chunks, embeddings)

# 대화를 텍스트 파일로 저장하는 함수
def save_conversation_as_txt(chat_history):
    conversation = ""
    for user_msg, ai_msg in chat_history:
        conversation += f"User: {user_msg}\nAI: {ai_msg}\n\n"
    
    b64 = base64.b64encode(conversation.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="대화.txt">대화 다운로드</a>'
    st.markdown(href, unsafe_allow_html=True)

# 애플리케이션 실행
if __name__ == '__main__':
    main()
