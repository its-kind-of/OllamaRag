from flask import Flask, request
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate

app = Flask(__name__)

folder_path = "db"

cached_llm = Ollama(model="llama3:latest")

embeddings = FastEmbedEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, )#is_separator_regex=True)

raw_prompt = PromptTemplate.from_template(""" 

   <s>[INST] You are an AI technical assistant good at searching documents. 
             You will be given a question. You must generate a detailed and long answer.
            if you don't have the answer from the provided information say so.
    [/INST]</s>?
    [INST] {input} 
            Context : {context}
            ANswer :
    [/INST]""")

@app.route("/ai", methods=["POST"])
def aiPost():
    print("Post/ai Called")
    json_content = request.json
    query = json_content.get("query")

    print("query: %s" % query)

    response = cached_llm.invoke(query)


    response_answer = {"answer":response}
    return response_answer

@app.route("/ask_pdf", methods=["POST"])
def askPdfPost():
    print("Post/ask_pdf Called")
    json_content = request.json
    query = json_content.get("query")

    print("query: %s" % query)

    print("loading vector store")
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embeddings)

    print("Creating chain")
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k":20, 
            "score_threshold": 0.1,
        },
    )

    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)

    response = chain.invoke({"input": query})

    sources = []

    for doc in response["context"]:
        sources.append(
            {"source": doc.metadata["source"], "page_content": doc.page_content}
        )


    response_answer = {"answer":response['answer'], "sources":sources}
    return response_answer


@app.route("/pdf", methods=["POST"])
def pdfPost():
    file = request.files['file']
    file_name = file.filename
    save_file = "pdf/" + file_name
    file.save(save_file)
    print(f"filename: %s" % file_name)

    loader = PDFPlumberLoader(save_file) 
    docs = loader.load_and_split()
    print(f"Doc len={len(docs)}")

    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=folder_path
        )

    vector_store.persist()


    response = {"status": "Successfully Uploaded", "filename": file_name, "doc len": len(docs), 
                "chunks len": len(chunks),}
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)