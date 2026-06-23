from dotenv import load_dotenv
from langchain_core.documents import Document
from typing import List, Dict, Optional
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.chat_history import(
    BaseChatMessageHistory,
    InMemoryChatMessageHistory
)
from datetime import datetime
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
load_dotenv()
import logging
from pydantic import BaseModel,Field
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import HumanMessage,AIMessage

logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
logging.getLogger("langchain_classic.retrievers.multi_query").setLevel(logging.INFO)

#要回答的像结构性的话
class ResearchResponse(BaseModel):
    answer:str=Field(description="The answer to the question")
    confidence:str=Field(description="high, medium, or low based on source quality")
    source:List[str]=Field(description="List of source documents used")
    key_quotes:List[str]=Field(
        description="Relevant quotes from sources",default=[])
    follow_up_questions:List[str]=Field(description="Suggested follow-up questions")
    
class AIResearchAssistant:
    def __init__(self,
                 persist_directory:str="./research_db",
                 chunk_size:int=1000,
                 chunk_overlap:int=200,):
        #数据库位置
        self.persist_directory=persist_directory
        #向量化工具：embeddings
        self.embedding=OpenAIEmbeddings(model="text-embedding-3-small")
        #llm
        self.llm=ChatOpenAI(model="gpt-4o-mini",temperature=0)
        #spliter
        self.splitter=RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        #vectorstore存上下文
        self.vectorstore=Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedding,
            collection_name="research_docs"
        )
        #Dic 存历史记录
        self.session_store:Dict[str,InMemoryChatMessageHistory]={}


        
    
    def add_text(self,text:str,source:str,metadata:dict=None)->int:
        doc=Document(
            page_content=text,
            metadata={"source":source,
                      **(metadata or {})}
        )
        return self.add_documents([doc])#source
    
    #TODO:add_texts,text是list[str]的情况
    
    #分割添加到数据库
    def add_documents(self,
                      docs:List[Document],
                      source_name:Optional[str]=None)->int:
        ##添加source 前面text添加了
        # for doc in docs:
        #     doc.metadata["source"]=source_name
            
        #分割成list[document]
        chunks=self.splitter.split_documents(docs)
        #添加时间戳给每个分好的块
        for chunk in chunks:
            chunk.metadata["indexed_at"]=datetime.now().isoformat()
        #添加到数据库中
        self.vectorstore.add_documents(chunks)
        print(f"Add{len(chunks)}chunks from {len(docs)} documents")
        return len(chunks)
    
    #创建基本retriever或者multiqueryretriever
    def build_retriever(self,advance:bool =False):
        baseretriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k":2})
       
        if not advance:
            return baseretriever

         #multiqueryretriever
        multiqueryretriever=MultiQueryRetriever.from_llm(
              llm=self.llm,
              retriever=baseretriever
          )
        return multiqueryretriever
    
    #把list[document]拆成字符串,它这里拆的时候是拆内容，然后source呢
    def format_docus_for_context(self,docs:List[Document])->str:
        if not docs:
            return "No relevant documents found."
        
        formatted=[]
        for i,doc in enumerate(docs):
            source=doc.metadata.get("source","Unknow")
            formatted.append(f"[Source{i+1}:{source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted) 
    

    #关于历史记录
    def get_session_history(self,session_id:str)->BaseChatMessageHistory:
        if(session_id not in self.session_store):
            self.session_store[session_id]=InMemoryChatMessageHistory()
        return self.session_store[session_id]
    
    #回答结构性
    def ask_structured(self,quesiotn:str,
                       session_id:str="default",
                       use_advanced:bool=True,):
        #构建prompt：input，模板，context，history
        prompt=ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an AI Research Assistant. Analyze the provided documents 
    and return a structured response.

    Rules:
    1. ONLY use information from the provided context
    2. If the context doesn't have the answer, say so in the answer field
    3. Set confidence: "high" if directly stated, "medium" if inferred, "low" if partial
    4. Include the source filenames you actually used
    5. Extract key quotes word-for-word from the context
    6. Suggest 2-3 follow-up questions the user might want to ask

    Use conversation history to understand follow-up questions.""",
                ),
        MessagesPlaceholder(variable_name="history"),
        ("human","""

        Context documents:

        {context}

        Available sources: {sources}

        Question: {question}

        """)
        ])
        
        #context
        retriever=self.build_retriever(use_advanced)
        docs=retriever.invoke(quesiotn)
        context=self.format_docus_for_context(docs)
        sources=list(set(d.metadata.get("source","Unknow") for d in docs))
        for s in sources:
            print(f"当前找到的source{s}")

        #history
        history=self.get_session_history(session_id)

        #structured_llm，再做chain之前
        structured_llm=self.llm.with_structured_output(ResearchResponse)
        chain=prompt|structured_llm
        response=chain.invoke(
            {"question":quesiotn,
             "context":context,
             "sources":", ".join(sources),
             "history":(history.messages[-10:] if hasattr(history,"messages") else history[-10:])
             }

        )
        history.add_message(HumanMessage(content=quesiotn))
        history.add_message(AIMessage(content=response.answer))
        return response


    
def print_research_response(question: str, response: ResearchResponse):
    """Pretty print a structured research response."""

    print(f"\nQ: {question}")
    print(f"\n  Answer: {response.answer}")
    print(f"\n  Confidence: {response.confidence}")
    print(f"  Sources: {', '.join(response.source)}")

    if response.key_quotes:
        print(f"\n  Key Quotes:")
        for q in response.key_quotes:
            print(f'    - "{q}"')

    print(f"\n  Follow-up Questions:")
    for fq in response.follow_up_questions:
        print(f"    - {fq}")

    
if __name__ == "__main__":
        print("开始运行agent")
import shutil

shutil.rmtree("./research_db", ignore_errors=True)
assistant = AIResearchAssistant()

    # Add research docs
assistant.add_text(
        """
        Attention Mechanisms in Neural Networks

        The attention mechanism was introduced in "Attention Is All You Need"
        by Vaswani et al. (2017). It allows models to focus on relevant parts
        of the input when generating output.

        Key concepts:
        - Query, Key, Value (QKV) triplets
        - Scaled dot-product attention
        - Multi-head attention for parallel processing

        The transformer architecture has become the foundation for modern NLP
        models including BERT, GPT, and T5.
        """,
        source="attention_mechanisms.pdf",
    )
assistant.add_text(
        """
        Retrieval-Augmented Generation (RAG)

        RAG combines retrieval systems with generative models. First introduced
        by Lewis et al. (2020), RAG addresses the limitation of LLMs being
        limited to their training data.

        Components of a RAG system:
        1. Document store with vector embeddings
        2. Retriever to find relevant documents
        3. Generator (LLM) to produce responses

        Benefits include reduced hallucination, up-to-date information,
        and source attribution.
        """,
        source="rag_survey.pdf",
    )

q1 = "What are the components of RAG?"
session = "structured_demo"
print(f"\nUser: {q1}")
r1 = assistant.ask_structured(q1, session)
print_research_response(q1, r1)


'''
没有相关文本的时候
Q: What are the components of RAG?

  Answer: The context does not provide information about the components of RAG (Retrieval-Augmented Generation).

  Confidence: low
  Sources: Source1:Unknow

  Follow-up Questions:
    - What is RAG and how does it work?
    - Can you explain the components of transformer models?
    - What are the applications of attention mechanisms in NLP?
'''

'''
有相关文本的时候
Q: What are the components of RAG?

  Answer: The components of a RAG system are: 1. Document store with vector embeddings 2. Retriever to find relevant documents 3. Generator (LLM) to produce responses.

  Confidence: high
  Sources: rag_survey.pdf

  Key Quotes:
    - ""Components of a RAG system: 1. Document store with vector embeddings 2. Retriever to find relevant documents 3. Generator (LLM) to produce responses.""

  Follow-up Questions:
    - What are the benefits of using RAG?
    - How does RAG reduce hallucination?
    - Can you explain how the retriever works in a RAG system?

'''

       
            

            
        
        
