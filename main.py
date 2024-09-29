import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
import re
from datetime import datetime

# Load environment variables (like OpenAI API keys, etc.)
load_dotenv()


# Function to retrieve programs based on query
def get_retriever(persist_directory="./db"):
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=OpenAIEmbeddings(model='text-embedding-3-small')
    )
    return vectorstore.as_retriever(search_kwargs={"k": 5})


# Function to initialize components and setup retriever and prompts
retriever = get_retriever()
llm = ChatOpenAI(model="gpt-4o-mini")


def initialize_components():
    today = datetime.today().strftime('%Y-%m-%d')
    # Define the contextualize question prompt
    contextualize_q_system_prompt = """
    #역할:
    채팅 기록과 질문을 토대로 대화 내역 없이도 이해할 수 있는 독립적인 질문을 생성하는 봇
    
    #지시사항:
    대화 내역과 최신 사용자 질문을 참고하여 질문 작성
    질문에 답변하지 말고, 필요하면 질문을 재구성한 후 그대로 반환
    '오늘', '이번 달', '지금', '이번주'과 같은 상대적인 날짜가 있다면, '오늘'을 기준으로 변경
    질문에 날짜가 언급되지 않았다면, '오늘'을 기준으로 최대 2달 전까지를 문의하세요.
    
    #오늘: {today}
    
    #예시1
    채팅기록:
    - 사용자: "2024년 10월 평창 여행지원금에 대해 알려줘"
    - 어시스턴트: "A, B, C 프로그램이 있습니다."
    - 사용자 : "그 중에서 아이와 함께 할 수 있는 프로그램이 뭐야?"
    
    답변: "A, B, C 프로그램 중에서 아이와 함께 할 수 있는 프로그램은 무엇인가요?"
    
    #예시2
    채팅기록:
    - 사용자: "전라도 여행지원금에 대해 알려줘"
    
    답변: "{today} 에 갈 수 있는 전라도 여행지원금에 대해 알려주세요."    
    """.format(today=today)

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # Define the answer question prompt
    qa_system_prompt = """
    #역할:
    질문에 대해 여행지원금 프로그램을 찾아주는 봇
    
    #지시사항:
    - 답변은 아래의 '답변 형식'에 맞추어 작성하십시오. 단, 프로그램에 대해 설명해달라는 질문이 있을 경우, 답변 형식을 무시하고 해당 프로그램에 대한 설명을 해주십시오.
    - 다음의 참고할 프로그램 정보를 사용하여 질문에 답하십시오.
    - 질문과 주어진 context 의 지역(광역시/도, 시/군/구)이 다른 경우 질문의 답으로 사용하지 마십시오.
    - 지역이 명시되지 않은 경우 모든 context 를 질문의 답으로 활용하십시오. 
    - 대답은 한국어로 하고, 존댓말을 사용하십시오.
    - 질문에 대한 답변이 있다면 답변의 마지막에 '자세한 사항은 링크를 통해 확인하세요' 문구와 이모지를 추가하세요.
    - 질문에 대한 답변이 없다면 답변의 마지막에 링크를 추가하지 마세요.
    
    #제약:
    - 특정 날짜가 나오면 활동 시작 시간 ~ 활동 종료 시간으로 판단하십시오.
    - '답변 형식1'은 최대 3개 프로그램으로 작성하십시오.
    - 연관된 프로그램이 2개 이상이라면 2개 이상 모두 작성하십시오.
    - 프로그램의 세부사항이 다르다면 해당 프로그램을 답변으로 사용하지 마십시오. 예를 들어서 아이와 갈 수 있는 프로그램을 물어봤을 때 '아이 동반 여부'는 가능이여야 합니다.
    - 알맞은 프로그램이 없는 경우, 해당하는 프로그램이 없다고 답변하십시오.
    
    #주요 key 해석
    활동 시작 시간: 여행, 활동을 시작하는 날짜입니다.
    활동 종료 시간: 여행, 활동을 종료하는 날짜입니다.
    최대 활동(여행)일수: 여행, 활동을 최대로 할 수 있는 일수입니다.
    최소 활동(여행)일수: 여행, 활동을 최소로 할 수 있는 일수입니다.
    공고(지원) 마감 시간: 사용자가 프로그램에 지원하는 마감 시간입니다.
    공고(지원) 시작 시간: 사용자가 프로그램에 지원하는 시작 시간입니다.
    아이 동반 가능(아이 함께 가기): 아이와 함께 여행을 할 수 있는지 여부입니다.
    아이 지원금 지원 여부: 프로그램에서 아이에게 지원금을 지원하는지 여부입니다.
    반려동물 동반 가능 여부: 반려동물을 동반할 수 있는지 여부입니다.
    유료 프로그램 여부: 프로그램이 유료인지 여부입니다. 유료인 경우, 참가비가 필요합니다.
    선착순 여부: 프로그램을 지원할 때 선착순인지 확인합니다.
    전액 지원 여부: 프로그램을 전액 지원하는지 여부입니다. 지원금 높은 순을 질문할 때 항상 전액 지원이 우선합니다.
   
    #답변 형식:\n
    - 1번 프로그램\n
      - 프로그램 제목: 00 한달살기\n
      - 프로그램 내용: 00에서 한달살기를 하는 프로그램입니다. 숙박비가 지원됩니다.\n
      - 활동 시작 시간: 2024-01-26\n
      - 활동 종료 시간: 2024-02-10\n
      - 공고(지원) 시작 시간: 2024-01-01\n
      - 공고(지원) 마감 시간: 2024-01-10\n
      - 한달살러 링크: https://www.monthler.kr/programs/12\n
    ---\n
    - 2번 프로그램\n
      - 프로그램 제목: 2024 00 스탬프 투어\n
      - 프로그램 내용: 00에서 스탬프 투어를 하고 지원금을 받으세요.\n
      - 활동 시작 시간: 2024-03-01\n
      - 활동 종료 시간: 2024-04-20\n
      - 공고(지원) 시작 시간: 2024-02-01\n
      - 공고(지원) 마감 시간: 2024-02-15\n
      - 한달살러 링크: https://www.monthler.kr/programs/672\n
    ---\n
    - 3번 프로그램\n
      - 프로그램 제목: 00에서 캠핑하자\n
      - 프로그램 내용: 00에서 캠핑을 하고 지원금을 받으세요.\n
      - 활동 시작 시간: 2024-05-01\n
      - 활동 종료 시간: 2024-05-10\n
      - 공고(지원) 시작 시간: 2024-04-01\n
      - 공고(지원) 마감 시간: 2024-04-15\n
      - 한달살러 링크: https://www.monthler.kr/programs/45\n
   
    #참고할 프로그램 정보:
    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


# Main function to handle user input and output
st.header("여행지원금 Q&A 챗봇 💬 📚")

rag_chain = initialize_components()
chat_history = StreamlitChatMessageHistory(key="chat_messages")

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",
)

# Initialize session state for messages
if "messages" not in st.session_state:
    st.chat_message("assistant").write("여행지원금을 검색하세요!")

# Display previous messages from session state
for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)

# Process user input
if prompt_message := st.chat_input("Your question"):
    # Store user's message in session state
    st.chat_message("human").write(prompt_message)

    # Perform search and display results
    with st.chat_message("assistant"):
        with st.spinner("검색 중..."):
            config = {"configurable": {"session_id": "any"}}
            response = conversational_rag_chain.invoke(
                {"input": prompt_message},
                config)

            answer = response['answer']
            st.write(answer)