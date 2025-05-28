from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

import os
import chainlit as cl
import PyPDF2
from io import BytesIO
from dotenv import load_dotenv
import unicodedata
import re
import logging
import json
from datetime import datetime
from pdf2image import convert_from_bytes
import pytesseract
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# lazy import SBERT
# lazy import spaCy

load_dotenv()

# Lazy loading para spaCy e SBERT
sbert_model = None
nlp = None

def get_sbert_model():
    global sbert_model
    if sbert_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            sbert_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
            logging.info("✅ SBERT carregado com sucesso (lazy).")
        except Exception as e:
            logging.error(f"❌ Erro ao carregar SBERT: {e}")
            sbert_model = None
    return sbert_model

def get_nlp_model():
    global nlp
    if nlp is None:
        try:
            import spacy
            nlp = spacy.load("pt_core_news_md")
            logging.info("✅ spaCy carregado com sucesso (lazy).")
        except Exception as e:
            logging.error(f"❌ Erro ao carregar spaCy: {e}")
            nlp = None
    return nlp

logging.basicConfig(level=logging.INFO)

logging.info("✅ Início do script após carregar .env")

# Modo de recursos pesados (para uso local)
FULL_MODE = os.getenv("FULL_MODE", "false").lower() == "true"

# NLP com fallback
try:
    if FULL_MODE:
        import spacy
        nlp = spacy.load("pt_core_news_md")
        logging.info("✅ spaCy carregado com sucesso.")
    else:
        nlp = None
        logging.info("🔧 spaCy não carregado (modo leve).")
except Exception as e:
    logging.warning(f"⚠️ Erro ao carregar spaCy: {e}")
    nlp = None
sbert_model = None  # Iniciado como None

def get_sbert_model():
    global sbert_model
    if sbert_model is None:
        try:
            print("🔄 Carregando SBERT sob demanda...")
            # sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # modelo leve compatível com Railway
            logging.info("✅ SBERT carregado com sucesso (lazy).")
        except Exception as e:
            logging.error(f"❌ Erro ao carregar SBERT: {e}")
            sbert_model = None
    return sbert_model


def log_similarity_scores(query: str, docs: list, scores: list):
    
    log_data = {
        "query": query,
        "top_results": [
            {
                "source": doc.metadata.get("source", "?"),
                "score": float(score),
                "preview": doc.page_content[:50] + "..."
            } for doc, score in zip(docs, scores)
        ]
    }
    logging.info(f"[SIMILARITY SCORES] {json.dumps(log_data, ensure_ascii=False)}")

CHAT_HISTORY_FILE = "chat_history.json"

extracted_metadata = {}

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=300, separators=["\n\n", "\n", " ", ""])

def save_chat_history(user_input, response):
    if FULL_MODE:
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "response": response
        }
        logging.info(f"[HISTÓRICO] {history_entry}")
    else:
        logging.info(f"[LOG] Pergunta: {user_input} | Resposta: {response}")

    # Exemplo de chamada
    if __name__ == "__main__":
        txt = "O autor requer a procedência do pedido."
        tipo = detect_document_type(txt)
        ents = extract_named_entities(txt)
        save_chat_history("Que tipo de documento é?", f"{tipo}, entidades: {ents}")


@cl.action_callback("load_chat_history")
async def load_chat_history(action):
    
    chat_id = action.payload
    
    history = load_chat_history_from_storage(chat_id)
    
    if history:
        
        await reset_user_session()
        
        
        for msg in history["messages"]:
            if msg["role"] == "user":
                await cl.Message(content=msg["content"], author="Usuário").send()
            else:
                await cl.Message(content=msg["content"]).send()
        
        await cl.Message(content="Histórico de conversa restaurado.").send()
    else:
        await cl.Message(content="Não foi possível carregar o histórico.").send()

def load_chat_history_from_storage(chat_id):
    
    try:
        with open(f"chat_history{chat_id}.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None

def process_pdf_with_hybrid_extraction(pdf_bytes: bytes) -> str:
    
    pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    full_text = ""
    extraction_methods = []
    
    for i, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text() or ""
        
        
        if not page_text.strip() or len(page_text.strip()) < 100:
            logging.info(f"[OCR] Aplicando OCR na página {i+1} devido a texto insuficiente")
            try:
                
                images = convert_from_bytes(pdf_bytes, first_page=i+1, last_page=i+1)
                if images:
                    image = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2GRAY)
                    image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)[1]
                    ocr_text = pytesseract.image_to_string(image, lang="por")
                    page_text = ocr_text
                    extraction_methods.append(f"Página {i+1}: OCR")
                else:
                    extraction_methods.append(f"Página {i+1}: Falha na conversão para imagem")
            except Exception as e:
                logging.error(f"[OCR ERROR] Página {i+1}: {e}")
                extraction_methods.append(f"Página {i+1}: Erro OCR")
        else:
            extraction_methods.append(f"Página {i+1}: PyPDF2")
        
        full_text += page_text + "\n\n"
    
    logging.info(f"[EXTRACTION METHODS] {'; '.join(extraction_methods)}")
    return full_text

def detect_document_type(text: str) -> str:

    keywords = {
        "petição_inicial": ["petição inicial", "autor requer", "dos pedidos", "dos fatos", "do direito", 
                           "deferimento", "termos em que", "pede deferimento"],
        "contestação": ["contestação", "preliminarmente", "mérito", "improcedente", "improcedência", 
                       "contesta", "contestar"],
        "laudo_pericial": ["laudo pericial", "perícia", "perito", "quesitos", "vistoria", "exame", 
                          "conclusão técnica", "metodologia"],
        "sentença": ["sentença", "julgo", "dispositivo", "condeno", "improcedente", "procedente", 
                    "fundamentação", "relatório", "isto posto"],
        "despacho": ["despacho", "intime-se", "cite-se", "certifique-se", "cumpra-se", "manifeste-se"],
        "acórdão": ["acórdão", "votação", "turma", "câmara", "relator", "revisor", "ementa"]
    }
    
    
    counts = {doc_type: 0 for doc_type in keywords}
    text_lower = text.lower()
    
    for doc_type, terms in keywords.items():
        for term in terms:
            counts[doc_type] += text_lower.count(term)
    
    
    doc_types = list(keywords.keys())
    
    # Combina com embeddings se modo completo estiver ativado
    if FULL_MODE:
        model = get_sbert_model()
        if not model:
            return max(counts, key=counts.get)
        doc_descriptions = [
        "Petição inicial com pedidos e fatos",
        "Contestação com argumentos de defesa",
        "Laudo pericial com análise técnica",
        "Sentença judicial com decisão",
        "Despacho com determinações processuais",
        "Acórdão com decisão colegiada"
    ]
        embeddings = model.encode([text_lower[:1000]] + doc_descriptions)
        # lógica reduzida
        return max(counts, key=counts.get)

    
    model = get_sbert_model()
    if not model:
        return max(counts, key=counts.get)
    text_embedding = model.encode([text_lower[:1000]])[0]
    desc_embeddings = get_sbert_model().encode(doc_descriptions)
    
    
    similarities = cosine_similarity([text_embedding], desc_embeddings)[0]
    
    
    combined_scores = {
        doc_type: (counts[doc_type] * 0.7) + (similarities[i] * 0.3)
        for i, doc_type in enumerate(doc_types)
    }
    
    most_likely_type = max(combined_scores, key=combined_scores.get)
    confidence = combined_scores[most_likely_type]
    
    logging.info(f"[DOCUMENT TYPE] Detectado: {most_likely_type} (confiança: {confidence:.2f})")
    
    return most_likely_type

def expand_question_for_legal_context(question: str) -> str:
    synonyms = {
        "autor": ["reclamante", "parte autora", "requerente", "demandante", "nome do autor", "quem é o autor"],
        "réu": ["reclamada", "empresa", "demandado", "parte ré", "nome do réu", "quem é o réu"],
        "advogado": ["procurador", "representante legal", "oab", "defensor", "advogado da parte"],
        "perito": ["especialista", "médico perito", "engenheiro", "assistente técnico"]
    }

    generic_terms = ["nome", "quem é", "qual o nome", "identificação"]

    expanded = question.lower()

    if any(term in expanded for term in generic_terms):
        
        expanded += " autor reclamante réu reclamada parte"


    for key, terms in synonyms.items():
        if key in expanded:
            expanded += " " + " ".join(terms)
    return expanded

def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ASCII', 'ignore').decode('utf-8')
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

def extract_text_with_ocr(pdf_bytes: bytes) -> str:
    images = convert_from_bytes(pdf_bytes)
    full_text = ""
    for img in images:
        image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)[1]
        text = pytesseract.image_to_string(image, lang="por")
        full_text += text + "\n"
    return full_text

def process_pdf_with_hybrid_extraction(pdf_bytes: bytes) -> str:
    
    pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    full_text = ""
    extraction_methods = []
    
    for i, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text() or ""
        
        
        if not page_text.strip() or len(page_text.strip()) < 100:
            logging.info(f"[OCR] Aplicando OCR na página {i+1} devido a texto insuficiente")
            try:
                
                images = convert_from_bytes(pdf_bytes, first_page=i+1, last_page=i+1)
                if images:
                    image = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2GRAY)
                    image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)[1]
                    ocr_text = pytesseract.image_to_string(image, lang="por")
                    page_text = ocr_text
                    extraction_methods.append(f"Página {i+1}: OCR")
                else:
                    extraction_methods.append(f"Página {i+1}: Falha na conversão para imagem")
            except Exception as e:
                logging.error(f"[OCR ERROR] Página {i+1}: {e}")
                extraction_methods.append(f"Página {i+1}: Erro OCR")
        else:
            extraction_methods.append(f"Página {i+1}: PyPDF2")
        
        full_text += page_text + "\n\n"
    
    logging.info(f"[EXTRACTION METHODS] {'; '.join(extraction_methods)}")
    return full_text

def extract_named_entities(text: str):
    metadata = {}
    if not nlp:
        return metadata
    doc = get_nlp_model()(text)
    for ent in doc.ents:
        if ent.label_ == "PER":
            metadata["pessoa"] = ent.text
    return metadata

def rerank_semantically(question: str, documents: list[Document]) -> list[Document]:
    
    
    expanded_question = expand_question_for_legal_context(question)
    
    model = get_sbert_model()
    if not model:
        logging.warning("⚠️ SBERT indisponível - ignorando reranking semântico.")
        return documents
    
    if "réu" in question.lower() or "reu" in question.lower():
        expanded_question += " reclamado demandado parte contrária"
    if "autor" in question.lower():
        expanded_question += " reclamante requerente parte autora"
    if "advogado" in question.lower():
        expanded_question += " procurador representante legal oab"
    
    
    doc_texts = [doc.page_content for doc in documents]
    model = get_sbert_model()
    if not model:
        return documents
    doc_embeddings = model.encode(doc_texts)
    question_embedding = model.encode([expanded_question])[0]
    
    
    scores = cosine_similarity([question_embedding], doc_embeddings)[0]
    
    
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    if max(scores) < 0.4:
        return []
    
    
    logging.info(f"[RERANKING] Pergunta expandida: {expanded_question}")
    for i, (doc, score) in enumerate(ranked[:3]):
        logging.info(f"[RERANKING] Doc {i+1}, Score: {score:.4f}, Preview: {doc.page_content[:80]}...")
        
    return [doc for doc, _ in ranked]


def extract_explicit_metadata(text: str):
    metadata = {}

    if not nlp:
        return metadata

    doc = get_nlp_model()(text)
    for ent in doc.ents:
        if ent.label_ == "PER" and any(term in ent.sent.text.lower() for term in ["juiz", "magistrado", "julgador"]):
            metadata["juiz"] = ent.text
        if ent.label_ == "LOC" and any(term in ent.sent.text.lower() for term in ["endereço", "localizado", "sede"]):
            metadata["endereco_relevante"] = ent.text
        if ent.label_ == "LAW" or any(term in ent.text.lower() for term in ["lei", "artigo", "decreto", "clt"]):
            if "leis_citadas" not in metadata:
                metadata["leis_citadas"] = []
            if ent.text not in metadata["leis_citadas"]:
                metadata["leis_citadas"].append(ent.text)

    if "leis_citadas" in metadata and isinstance(metadata["leis_citadas"], list):
        metadata["leis_citadas"] = ", ".join(metadata["leis_citadas"])

    return metadata

def format_metadata_for_prompt(metadata: dict) -> str:
    if not metadata:
        return "Nenhum metadado detectado."
    return "\n".join([f"{k.replace('_', ' ').capitalize()}: {v}" for k, v in metadata.items()])


def build_adaptive_prompt(query: str, metadata: dict):
    
    metadata_str = format_metadata_for_prompt(metadata)
    
    
    query_lower = query.lower()
    
    
    specific_instructions = ""
    
    if any(term in query_lower for term in ["autor", "reclamante", "requerente", "parte"]):
        specific_instructions = """
        Ao responder sobre partes do processo:
        - Se a informação exata **não estiver disponível**, mas houver **qualquer menção relacionada**, **nunca responda apenas "informação não encontrada"**. Explique, com base no documento, o que é mencionado sobre o tema da pergunta.
        - Forneça apenas nomes completos, sem explicações adicionais
        - Se houver qualificação como CPF ou RG, inclua apenas se explicitamente solicitado
        - Seja extremamente conciso
        """
    elif any(term in query_lower for term in ["advogado", "procurador", "representante", "oab"]):
        specific_instructions = """
        Ao responder sobre representantes legais:
        - Forneça apenas o nome e número da OAB, sem explicações adicionais
        - Indique apenas a qual parte o advogado está vinculado, se necessário
        - Seja extremamente conciso
        """
    elif any(term in query_lower for term in ["data", "prazo", "audiência", "perícia"]):
        specific_instructions = """
        Ao responder sobre datas, prazos e perícias:
        - Se a informação exata não estiver disponível, explique o que o documento menciona sobre o assunto
        - Informe sobre determinações, procedimentos ou instruções relacionadas no documento
        - Cite trechos relevantes que mencionem como a informação será definida ou comunicada
        - Seja claro e informativo, mesmo quando a resposta direta não estiver presente
        """
    elif any(term in query_lower for term in ["valor", "causa", "condenação", "indenização", "dano"]):
        specific_instructions = """
        Ao responder sobre valores monetários:
        - Forneça apenas o valor e a que se refere
        - Não inclua explicações sobre juros ou correção, a menos que explicitamente solicitado
        - Seja extremamente conciso
        """
    
    system_template = f"""
    Você é um assistente especializado em análise pericial de documentos jurídicos. Sua tarefa é extrair TODAS as informações relevantes para um perito judicial que precisa realizar uma perícia com base neste documento.
    
    Analise o documento minuciosamente e extraia as seguintes categorias de informações:
    
    1. IDENTIFICAÇÃO DO PROCESSO:
       - Número do processo
       - Vara/Tribunal
       - Tipo de ação
       - Objeto da perícia
    
    2. PARTES ENVOLVIDAS:
       - Nome completo e qualificação do(a) reclamante/autor(a)
       - Nome completo e qualificação do(a) reclamado(a)/réu
       - Representantes legais (advogados) de cada parte com OAB
    
    3. PERÍCIA DETERMINADA:
       - Tipo específico de perícia solicitada
       - Objetivo da perícia
       - Especialidade requerida do perito
    
    4. PERITO JUDICIAL:
       - Nome completo do perito nomeado
       - Especialidade/qualificação do perito
       - Contatos do perito (se disponíveis)
    
    5. PRAZOS E DATAS:
       - Prazo para realização da perícia
       - Prazo para entrega do laudo
       - Prazo para manifestação das partes
       - Datas de audiências relacionadas
    
    6. QUESITOS E PONTOS DE INVESTIGAÇÃO:
       - Quesitos apresentados pelas partes
       - Pontos específicos determinados pelo juiz
       - Questões técnicas a serem respondidas
    
    7. DOCUMENTOS E EVIDÊNCIAS:
       - Documentos que devem ser analisados
       - Exames, laudos ou relatórios mencionados
       - Provas técnicas disponíveis
    
    8. PROCEDIMENTOS ESPECÍFICOS:
       - Metodologia determinada para a perícia
       - Locais a serem vistoriados
       - Pessoas a serem entrevistadas/examinadas
    
    9. HONORÁRIOS PERICIAIS:
       - Valor dos honorários
       - Responsável pelo pagamento
       - Forma e prazo de pagamento
    
    10. ASSISTENTES TÉCNICOS:
        - Prazo para indicação
        - Nomes dos assistentes já indicados
        - Regras para participação dos assistentes
    
    11. OUTRAS INFORMAÇÕES RELEVANTES:
        - Determinações judiciais específicas
        - Histórico relevante do processo
        - Informações sobre tentativas anteriores de perícia
    
    Para cada categoria, forneça TODAS as informações disponíveis no documento, mesmo que parciais. Se alguma informação não estiver explicitamente mencionada, indique claramente.
    
    Organize sua resposta em tópicos claros e objetivos, usando a estrutura acima. Inclua citações textuais relevantes quando necessário.

    {specific_instructions}

    Metadados extraídos automaticamente do documento:
    {metadata_str}
    ----------------
    {{context}}
    """
    
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ])


async def show_extracted_metadata(metadata: dict):
    if metadata:
        lines = ["| Campo | Valor |", "|-------|-------|"]
        lines += [f"| {k.replace('_', ' ').capitalize()} | {v} |" for k, v in metadata.items()]
        await cl.Message(content="🗂️ Metadados extraídos automaticamente do documento:\n" + "\n".join(lines)).send()
    else:
        await cl.Message(content="⚠️ Nenhum metadado foi detectado automaticamente.").send()


chain_type_kwargs = {
    "prompt": build_adaptive_prompt(query="", metadata={})  
}


async def reset_user_session():
    keys = ["chain", "retriever", "original_texts", "normalized_texts", "metadatas", "metadata"]
    for key in keys:
        if cl.user_session.get(key) is not None:
            cl.user_session.set(key, None)
            
def extract_pericial_metadata(text: str):
    """
    Extrai metadados específicos para perícias judiciais.
    """
    metadata = {}
    
    # Padrões de expressões regulares para informações periciais
    patterns = {
        "perito_nome": r"(?:perito|expert|especialista)[^\n.]*(?:nomeado|designado|indicado)[^\n.]*(?:Dr\.|Dr\(a\)\.?|Doutor|Dra\.|Doutora)?\s+([A-Z][a-zÀ-ú]+(?: [A-Z][a-zÀ-ú]+){1,5})",
        "prazo_pericia": r"(?:prazo|período)[^\n.]*(?:realização|realizar|efetuar)[^\n.]*(?:perícia|vistoria|exame)[^\n.]*((?:\d{1,2}|trinta|vinte|quinze|dez|cinco) (?:dias|meses))",
        "prazo_laudo": r"(?:prazo|período)[^\n.]*(?:entrega|apresentação|entregar|apresentar)[^\n.]*(?:laudo|relatório|parecer)[^\n.]*((?:\d{1,2}|trinta|vinte|quinze|dez|cinco) (?:dias|meses))",
        "tipo_pericia": r"(?:perícia|vistoria|exame)[^\n.]*(?:médica|médico|técnica|técnico|contábil|engenharia|ambiental|grafotécnica|psicológica|psiquiátrica)",
        "honorarios": r"(?:honorários|remuneração)[^\n.]*(?:periciais|do perito|da perícia)[^\n.]*R\$\s*(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)",
    }
    
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            metadata[key] = matches[0]
    
    # Extração de quesitos (mais complexa)
    quesitos_pattern = r"(?:QUESITOS|Quesitos|quesitos)(?:[^\n]*\n){1,3}((?:(?:\d+[\)\.-]|\-|\*)[^\n]+\n?)+)"
    quesitos_matches = re.findall(quesitos_pattern, text)
    if quesitos_matches:
        metadata["quesitos"] = quesitos_matches[0]
    
    # Adicionar ao extract_explicit_metadata existente
    if nlp:
        doc = get_nlp_model()(text)
        for ent in doc.ents:
            # Identificar especialidades médicas/técnicas
            if ent.label_ == "MISC" and any(term in ent.text.lower() for term in ["ortopedia", "cardiologia", "neurologia", "psiquiatria", "engenharia"]):
                metadata["especialidade_pericial"] = ent.text
    
    return metadata

@cl.on_chat_start
async def on_chat_start():
    elements = [cl.Image(name="image1", display="inline", path="image1PeritoDoc.jpg")]
    await cl.Message(content="Olá! Bem-vindo ao Chat Pericial! Envie um PDF para começar. 🤖", elements=elements).send()

    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Por favor, envie um arquivo PDF para começarmos.",
            accept=["application/pdf"],
            max_size_mb=60,
            timeout=240,
        ).send()

    file = files[0]
    msg = cl.Message(content=f"Processando `{file.name}`...")
    await msg.send()

    try:
        with open(file.path, "rb") as f:
            pdf_bytes = f.read()
        
        
        pdf_text = process_pdf_with_hybrid_extraction(pdf_bytes)
        source_method = "Híbrido"
        
        if not pdf_text.strip():
            raise ValueError("Não foi possível extrair texto do documento.")
    except Exception as e:
        logging.error(f"[PDF EXTRACTION ERROR] {e}")
        await cl.Message(content=f"Erro ao processar o PDF: {str(e)}").send()
        return

    logging.info("[DOCUMENT TYPE] Detecção desativada para teste")

        
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    global extracted_metadata
    extracted_metadata = extract_explicit_metadata(pdf_text)
    
    # Adicionar metadados periciais específicos
    pericial_metadata = extract_pericial_metadata(pdf_text)
    extracted_metadata.update(pericial_metadata)
    
    logging.info(f"[METADATA EXTRAIDA] {extracted_metadata}")

    pdf_text = pdf_text.replace("-\n", "").replace("\n", " ")
    original_chunks = text_splitter.split_text(pdf_text)

    
    for i, chunk in enumerate(original_chunks):
        if any(kw in chunk.lower() for kw in ["deverá informar", "com antecedência de", "designar perícia", "será designada", "intimar para perícia"]):
            original_chunks[i] = "[INSTRUÇÃO FUTURA] " + chunk

    normalized_texts = [normalize_text(t) for t in original_chunks]

    metadatas = [{"source": f"Trecho {i+1}"} for i in range(len(normalized_texts))]

    logging.info(f"[EXTRACTION] Método: {source_method}, Chunks gerados: {len(original_chunks)}")

    
    await reset_user_session()
        
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    docsearch = await cl.make_async(FAISS.from_texts)(
        normalized_texts, embeddings, metadatas=metadatas
    )
    
    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 12}  
    )

    
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-4", temperature=0),
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={
            "prompt": build_adaptive_prompt(query="", metadata=extracted_metadata),
            "document_variable_name": "context"
        }
    )

    
    cl.user_session.set("chain", chain)
    cl.user_session.set("retriever", retriever)
    cl.user_session.set("original_texts", original_chunks)
    cl.user_session.set("normalized_texts", normalized_texts)
    cl.user_session.set("metadatas", metadatas)
    cl.user_session.set("memory", memory)

    msg.content = f"Processamento de `{file.name}` concluído com sucesso via `{source_method}`! Pode perguntar algo. 📄"
    await msg.update()

@cl.on_message
async def main(message: str):
    chain = cl.user_session.get("chain")
    if chain is None:
        await cl.Message(content="⚠️ Cadeia não inicializada. Envie um PDF para começar.").send()
        return

    
    logging.info(f"[PERGUNTA ORIGINAL] {message.content}")
    
    is_pericial_extraction = any(term in message.content.upper() for term in [
        "INFORMAÇÕES RELEVANTES PARA UM AGENTE PERICIAL", 
        "INFORMAÇÕES PARA PERITO",
        "EXTRAÇÃO PERICIAL",
        "DADOS PERICIAIS",
        "INFORMAÇÕES PERICIAIS",
        "RELATÓRIO PERICIAL"
    ])

    query = normalize_text(message.content)
    docs = await chain.retriever.ainvoke(query)

        
    logging.info(f"[DOCUMENTOS RECUPERADOS] Total: {len(docs)}")
    for i, doc in enumerate(docs[:3]):
        logging.info(f"[DOC {i+1}] Fonte: {doc.metadata.get('source', '?')}")
        logging.info(f"[DOC {i+1}] Conteúdo: {doc.page_content[:150]}...")

    docs = rerank_semantically(message.content, docs)
    
    if is_pericial_extraction:
        # Usar mais documentos para extração pericial completa
        docs = await chain.retriever.ainvoke(query)
        # Limitar a um número razoável, mas maior que o padrão
        if len(docs) > 8:
            docs = docs[:8]
        logging.info(f"[EXTRAÇÃO PERICIAL] Usando {len(docs)} documentos para análise completa")
    else:
        # Para consultas normais, manter o reranking semântico
        docs = rerank_semantically(message.content, docs)

    if not docs:
        query_terms = " ".join([word for word in message.content.lower().split() if len(word) > 3])
        fallback_docs = await chain.retriever.ainvoke(query_terms)
        
        if fallback_docs:
            docs = fallback_docs[:3]  
            logging.info(f"[FALLBACK] Usando busca alternativa com termos: {query_terms}")
        else:
            await cl.Message(content="Não foi possível encontrar informações relevantes no documento para responder sua pergunta.").send()
            return

    try:
        # Escolher o prompt adequado com base no tipo de consulta
        if is_pericial_extraction:
            prompt = build_adaptive_prompt(extracted_metadata)
            logging.info("[PROMPT] Usando prompt especializado para extração pericial completa")
        else:
            prompt = build_adaptive_prompt(message.content, extracted_metadata)
            logging.info("[PROMPT] Usando prompt adaptativo padrão")
        
        chain.combine_docs_chain.llm_chain.prompt = prompt

        if not is_pericial_extraction and len(docs) > 3:
            docs = docs[:3]
            
        logging.info("[TRECHOS USADOS] " + "; ".join([doc.metadata.get("source", "?") for doc in docs]))
        
        start_time = datetime.now()
        logging.info(f"[PERGUNTA RECEBIDA] {message.content}")
        
                # Para extração pericial, aumentamos o tempo máximo de resposta
        if is_pericial_extraction:
            # Informar ao usuário que a análise completa está em andamento
            await cl.Message(content="🔍 Realizando análise pericial completa do documento. Isso pode levar alguns instantes...").send()
            
        res = await chain.ainvoke({"question": message.content})
        response_time = (datetime.now() - start_time).total_seconds()
        logging.info(f"[TEMPO DE RESPOSTA] {response_time:.2f} segundos")

        if isinstance(res, dict):
            answer = res.get("answer") or "Resposta não encontrada."
            
            # Para extração pericial, não limitamos o tamanho da resposta
            if not is_pericial_extraction and len(answer) > 2000:
                answer = answer[:1997] + "..."
            
            save_chat_history(message.content, answer)
            
            if is_pericial_extraction:
                    await cl.Message(content="## 📋 RELATÓRIO DE EXTRAÇÃO PERICIAL\n\n" + answer).send()
            else:
                await cl.Message(content=answer).send()
        else:
            answer = str(res)
            if not is_pericial_extraction and len(answer) > 2000:
                answer = answer[:1997] + "..."
            await cl.Message(content=answer.strip()).send()

        save_chat_history(message.content, answer)
        logging.info(f"[PERGUNTA] {message.content}")
        logging.info(f"[RESPOSTA] {answer[:600]}...")

    except Exception as e:
        logging.error(f"[LLM ERROR] {e}")
        answer = f"Erro ao gerar resposta: {str(e)}"
        await cl.Message(content=answer.strip()).send()
