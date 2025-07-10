import json
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

def load_data(filePath):
    with open(filePath, 'r')as file:
        return json.load(file)
      
def process_document(policies):
    documents = []
    for policy in policies:
        content = f"Title: {policy['title']}\nDescription: {policy['description']}\nType: {policy['type']}"
        metadata = {"title": policy["title"], "type": policy["type"]}
        documents.append(Document(page_content=content,metadata=metadata))
    return documents


def create_retriever():
    #data required
    raw_data = load_data('data.json')
    documents = process_document(raw_data)
    
    
    #Initialize embedding model
    embedding = HuggingFaceEmbeddings(
		model_name="all-MiniLM-L6-v2",
		model_kwargs={'device':'cpu'}
	  )
    
    
    #Initialize vectorStore
    vectorstore = Chroma.from_documents(
		embedding=embedding,
        documents=documents,
        collection_name='Leave_policy_school'
	)
    
    
    #Create and return retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k":3})
    
    retriever.add_documents(documents)
    return retriever
  
retriever = create_retriever()

def get_context(question):
    
    docs = retriever.get_relevant_documents(question)
    
    if not docs:
        return ""
    
    context = "\n\n".join(doc.page_content for doc in docs)
    return context
    