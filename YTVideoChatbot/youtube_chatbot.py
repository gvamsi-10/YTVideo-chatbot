from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser



import streamlit as st

import os
os.environ["HF_TOKEN"] = "hf_ZvoKXmSOeAoCvWXcK"

###STEP-1 INDEXING(1.1document ingestion)###
st.title("Youtube-video Chat App")
st.write("An ai chatbot to get more information about the youtube-video")

video_id = st.text_input('Video-Id', placeholder="Ex:3R9t3nHSI")
yta = YouTubeTranscriptApi();

st.session_state.transcript = None

if st.button("Load Content"):
    if not video_id:
        st.warning("please enter a video id")
    else:
        try:
            with st.spinner("Fetching Transcript..."):
                transcript = yta.fetch(video_id, languages=['en'])
            transcript_text = " ".join([entry.text for entry in transcript])
            st.session_state.transcript = transcript_text
            st.success("Transcript loaded successfully")
        except Exception:
            st.error("Couldn't load the content of provided video id")
            st.warning(""" Possible reasons:
            Invalid video id, Transcript is disabled, video is private, video has no captions, captions are in other than english""")
            st.stop()
###STEP-1.2 TEXT SPLITTING)###
    with st.spinner("Preparing the chatbot... Please wait"):
        splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
        chunks = splitter.create_documents([transcript_text])

# print(chunks[10])

###STEP-1.3 EMBEDDING###
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embedding)
# print(vector_store.index_to_docstore_id)

###STEP-2 RETRIEVER###

        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})


####MODEL INVOKING#######
        llm = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-7B-Instruct",
            task="text-generation"
        )

        model = ChatHuggingFace(llm=llm)

###STEP-3 AUGUMENTATION###

        prompt = PromptTemplate(
            template = """
            Act as an helpful assistance,
            Answer the questions only from the provided context,
            If the context is insufficient just say you dont know the answer.
            {context}
            Question: {question}
            """,
            input_variables=["context","question"]
        )
    
        if st.session_state.transcript is not None:

# question = ""
# retrieved_docs = retriever.invoke(question)
# context = "\n\n".join(doc.page_content for doc in retrieved_docs)
# final_prompt = prompt.invoke({'context':context,'question':question})


###STEP-4 GENERATION###

# answer = model.invoke(final_prompt)


########IMPLEMENTATION USING CHINS###########

            def format_docs(retrieved_docs):
                context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
                return context_text
            
            
            parser = StrOutputParser()

            paralle_chain = RunnableParallel({
                'context': retriever | RunnableLambda(format_docs),
                'question': RunnablePassthrough()
            })

            merge_chain = prompt | model | parser


            chain = paralle_chain | merge_chain
            
            st.session_state.chain = chain
        st.success("Chatbot is ready")

if "chain" in st.session_state:
    question = st.chat_input("Ask your question about this video")
    if question:
        with st.chat_message("user:"):
            st.write("User: ", question)

        with st.spinner("Thinking.."):
            answer = st.session_state.chain.invoke(question)

        with st.chat_message("Assistant"):
            st.write(answer)
