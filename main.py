# 라이브러리 및 모듈 가져오기
import streamlit as st
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

# 최신 구조: schema → core
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document as CoreDocument  # 더미 문서용

import tiktoken
import base64
import speech_recognition as sr
import tempfile

# 애플리케이션 실행 함수 정의
def main():
    st.set_page_config(page_title="에너지", page_icon="🌻", layout="centered")
    st.image('energy.png')
    st.title("_:red[에너지 학습 도우미]_ 🏫")
    st.header("😶주의! 이 챗봇은 참고용으로 사용하세요!", divider='rainbow')

    # 세션 상태
    if "conversation" not in st.session_state:
        st.session_state.conversation = None  # LCEL 체인
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []    # [HumanMessage, AIMessage]
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None
    if "voice_input" not in st.session_state:
        st.session_state.voice_input = ""
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "😊"}]

    # 사이드바
    with st.sidebar:
        folder_path = Path()
        if "OPENAI_API_KEY" not in st.secrets:
            st.error("secrets에 OPENAI_API_KEY가 없습니다. .streamlit/secrets.toml에 설정해 주세요.")
            st.stop()
        openai_api_key = st.secrets["OPENAI_API_KEY"]
        model_name = "gpt-4o-mini"

        st.text("아래의 'Process'를 누르고\n아래 채팅창이 활성화 될 때까지\n잠시 기다리세요!😊😊😊")
        process = st.button("Process", key="process_button")

        if process:
            try:
                files_text = get_text_from_folder(folder_path)
                text_chunks = get_text_chunks(files_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = build_lcel_chain(vectorstore, openai_api_key, model_name)
                st.session_state.processComplete = True
                st.success("인덱스가 준비되었습니다. 이제 질문하세요!")
            except Exception as e:
                st.exception(e)
                st.error("인덱스 준비 중 오류가 발생했습니다. 로그를 확인하세요.")
                st.stop()

        # 음성 입력
        audio_value = st.audio_input("음성 메시지를 녹음하여 질문하세요😁.")
        if audio_value:
            with st.spinner("음성을 인식하는 중..."):
                recognizer = sr.Recognizer()
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                        temp_audio_file.write(audio_value.getvalue())
                        with sr.AudioFile(temp_audio_file.name) as source:
                            audio = recognizer.record(source)
                            st.session_state.voice_input = recognizer.recognize_google(audio, language='ko-KR').strip()
                except sr.UnknownValueError:
                    st.warning("음성을 인식하지 못했거나 모델을 불러오지 않았습니다. Process를 누르고 다시 시도하세요!")
                except sr.RequestError:
                    st.warning("서버와의 연결에 문제가 있습니다. 다시 시도하세요!")
                except OSError:
                    st.error("오디오 파일을 처리하는 데 문제가 발생했습니다. 다시 시도하세요!")

        # 대화 저장
        if st.button("대화 저장", key="save_button"):
            if st.session_state.chat_history:
                save_conversation_as_txt(st.session_state.chat_history)
            else:
                st.warning("질문을 입력받고 응답을 확인하세요!")

        # 초기화
        if st.button("대화 내용 삭제", key="clear_button"):
            st.session_state.chat_history = []
            st.session_state.messages = [{"role": "assistant", "content": "😊"}]
            st.query_params

    # 입력
    query = st.session_state.voice_input or st.chat_input("질문을 입력해주세요.")

    if query:
        st.session_state.voice_input = ""
        try:
            st.session_state.messages.insert(0, {"role": "user", "content": query})
            chain = st.session_state.conversation
            with st.spinner("생각 중..."):
                if chain:
                    result = chain.invoke({"input": query, "chat_history": st.session_state.chat_history})
                    response = result.get("answer", "")
                    source_documents = result.get("context", [])
                    # 이력 업데이트
                    st.session_state.chat_history.append(HumanMessage(content=query))
                    st.session_state.chat_history.append(AIMessage(content=response))
                else:
                    response = "모델이 준비되지 않았습니다. 'Process' 버튼을 눌러 모델을 준비해주세요."
                    source_documents = []
        except Exception as e:
            st.exception(e)
            st.error("질문을 처리하는 중 오류가 발생했습니다. 위 로그를 확인하세요.")
            response, source_documents = "", []

        st.session_state.messages.insert(1, {"role": "assistant", "content": response})

    # 대화 표시
    for message_pair in list(zip(st.session_state.messages[::2], st.session_state.messages[1::2])):
        with st.chat_message(message_pair[0]["role"]):
            st.markdown(message_pair[0]["content"])
        with st.chat_message(message_pair[1]["role"]):
            st.markdown(message_pair[1]["content"])
        if 'source_documents' in locals() and source_documents:
            with st.expander("참고 문서 확인"):
                for doc in source_documents:
                    st.markdown(doc.metadata.get('source', '출처 없음'), help=getattr(doc, "page_content", ""))

# 토큰 길이 계산
def tiktoken_len(text: str) -> int:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(text))

# 폴더에서 문서 로드
def get_text_from_folder(folder_path: Path):
    doc_list = []
    folder = Path(folder_path)
    if not folder.exists():
        return doc_list
    for file in folder.iterdir():
        if file.is_file():
            if file.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file))
                documents = loader.load_and_split()
            elif file.suffix.lower() == ".docx":
                loader = Docx2txtLoader(str(file))
                documents = loader.load_and_split()
            elif file.suffix.lower() == ".pptx":
                loader = UnstructuredPowerPointLoader(str(file))
                documents = loader.load_and_split()
            else:
                documents = []
            doc_list.extend(documents)
    return doc_list

# 청크 분할
def get_text_chunks(text_docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    if not text_docs:
        return []
    return splitter.split_documents(text_docs)

# 벡터 스토어
def get_vectorstore(text_chunks):
    if not text_chunks:
        # 빈 인덱스 방지용 더미 문서 (core 문서 타입 사용)
        text_chunks = [CoreDocument(page_content="(no documents indexed)", metadata={"source": "none"})]
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return FAISS.from_documents(text_chunks, embeddings)

# ✅ LCEL 전용 History-Aware RAG (langchain.chains 없이 구성)
def build_lcel_chain(vectorstore, openai_api_key: str, model_name: str):
    # 최신 langchain_openai는 model 파라미터 사용
    llm = ChatOpenAI(openai_api_key=openai_api_key, model=model_name, temperature=0)

    # 1) 히스토리 기반 질문 재작성 → standalone_question
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the user's question into a standalone query for retrieval, considering the chat history."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    question_gen = rewrite_prompt | llm | StrOutputParser()

    retriever = vectorstore.as_retriever()  # 버전 호환 위해 기본값 사용

    # 2) 검색(LCEL retriever.invoke)
    def retrieve_docs(inputs):
        standalone_q = inputs["standalone_question"]
        return retriever.invoke(standalone_q)

    # 3) 답변 프롬프트 (문맥은 문자열로)
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer the user's question using ONLY the provided context. "
         "If the context is insufficient, say you don't know.\n\nContext:\n{context_str}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    answer_chain = answer_prompt | llm | StrOutputParser()

    # 4) 전체 체인 조립
    from langchain_core.runnables import RunnableMap

    def join_docs_as_text(docs):
        return "\n\n".join([getattr(d, "page_content", "") for d in docs]) if docs else "(no context)"

    chain = (
        # 입력 정규화
        RunnableMap({
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"],
        })
        # 질문 재작성
        | RunnableMap({
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"],
            "standalone_question": question_gen,
        })
        # ── 단계 1: 문서 검색 결과 생성
        | RunnableMap({
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"],
            "standalone_question": lambda x: x["standalone_question"],
            "context_docs": retrieve_docs,  # list[Document]
        })
        # ── 단계 2: 앞 단계의 context_docs를 문자열로 변환
        | RunnableMap({
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"],
            "context_docs": lambda x: x["context_docs"],
            "context_str": lambda x: join_docs_as_text(x["context_docs"]),
        })
        # 답변 생성 + UI용 컨텍스트 유지
        | RunnableMap({
            "answer": answer_chain,                 # string
            "context": lambda x: x["context_docs"], # list[Document]
        })
        # 출력 표준화
        | (lambda x: {"answer": x["answer"], "context": x["context"]})
    )
    return chain

# 대화 저장
def save_conversation_as_txt(chat_history):
    conversation = ""
    for message in chat_history:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        conversation += f"유저: {role}\n내용: {message.content}\n\n"
    b64 = base64.b64encode(conversation.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="대화.txt">대화 다운로드</a>'
    st.markdown(href, unsafe_allow_html=True)

# 실행
if __name__ == '__main__':
    main()
