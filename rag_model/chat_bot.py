#pdf loader 
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

loader = PyPDFLoader("rag_model/budget.pdf")
docs = loader.load()

# print(docs[0].page_content)

model = init_chat_model("openai/gpt-oss-120b",
                        model_provider="groq",
                        )
template = ChatPromptTemplate.from_messages(
    [("system","""
You are an intelligent AI assistant designed to answer questions strictly based on the provided context.

Rules:
1. Only use the information given in the context.
2. Do NOT use your own knowledge.
3. If the answer is not present in the context, say:
   "The answer is not available in the provided document."
4. Provide clear, concise, and accurate answers.
5. If possible, summarize relevant parts from the context.
6. Do not make assumptions or fabricate information.
"""),("human","""
Context:
{context}

Question:
{question}

Answer:
""")
]
)
question = input("Write your question-->")
context = "\n\n".join([doc.page_content for doc in docs[0:3]])
prompt = template.format_messages(context = context, 
                                question = question)

response = model.invoke(prompt)
print(response.content)