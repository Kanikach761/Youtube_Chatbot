
import os
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

# Set Google API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in environment variables. Set it before running.")
    st.stop()

st.set_page_config(page_title="YouTube Gemini Chatbot", layout="centered")
st.title("🎥 YouTube Gemini Chatbot")

video_url = st.text_input("Enter YouTube Video URL:", "")
if video_url:
    try:
        video_id = video_url.split("v=")[-1].split("&")[0]
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)

        # Show transcript preview
        with st.expander("📜 Transcript Preview"):
            st.write(transcript[:1000] + "...")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_document")
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro", temperature=0.2)

        prompt = PromptTemplate(
            template="""
            You are a helpful assistant.
            Answer ONLY from the provided transcript context.
            If the context is insufficient, just say you don't know.

            {context}
            Question: {question}
            """,
            input_variables=['context', 'question']
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        parallel_chain = RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })

        main_chain = parallel_chain | prompt | llm | StrOutputParser()

        st.subheader("💬 Ask questions about the video")
        user_question = st.text_input("Your question:")
        if user_question:
            with st.spinner("Thinking..."):
                response = main_chain.invoke(user_question)
                st.success(response)

        if st.button("📄 Summarize Video"):
            with st.spinner("Summarizing..."):
                summary = main_chain.invoke("Summarize the entire video.")
                st.info(summary)

    except Exception as e:
        st.error(f"Error: {str(e)}")
