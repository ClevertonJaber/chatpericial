from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
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
from chainlit.element import Text
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

def log_similarity_scores(query: str, docs: list, scores: list):
    """Registra os scores de similaridade dos documentos recuperados"""
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

# Metadados extraídos manualmente
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
        
        
        if len(page_text.strip()) < 50 and len(re.findall(r'\w+', page_text)) < 10:
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
    """Detecta o tipo de documento jurídico com base no conteúdo"""
    # Palavras-chave para cada tipo de documento
    keywords = {
        "petição_inicial": ["petição inicial", "autor requer", "dos pedidos", "dos fatos", "do direito", 
                           "deferimento", "termos em que", "pede deferimento", "ação", "proposta"],
        "contestação": ["contestação", "preliminarmente", "mérito", "improcedente", "improcedência", 
                       "contesta", "contestar", "defesa", "resposta", "exceção", "impugnação", "contestando"],
        "recurso": ["recurso", "recorre", "reforma", "reformar", "decisão recorrida", "data venia",
                   "colenda", "egrégio", "recurso ordinário", "agravo de instrumento", "embargos"],
        "sentença": ["sentença", "julgo", "dispositivo", "condeno", "improcedente", "procedente", 
                    "fundamentação", "relatório", "isto posto", "decido", "sentença de mérito"],
        "ctps": ["carteira de trabalho", "ctps", "anotação", "registro de empregado", "admissão", "demissão", "registro"],
        "holerite": ["contracheque", "holerite", "folha de pagamento", "salário", "remuneração", 
                    "proventos", "descontos", "líquido a receber", "total bruto", "base de cálculo"],
        "contrato_trabalho": ["contrato de trabalho", "contrato individual", "prazo indeterminado", 
                             "regime de trabalho", "jornada", "remuneração", "contratação", "empregador", "empregado"],
        "cnis": ["cnis", "cadastro nacional", "vínculos", "contribuições", "extrato previdenciário",
                "inss", "nis", "pis/pasep", "histórico de contribuições", "tempo de serviço"],
        "ppp": ["perfil profissiográfico", "ppp", "agentes nocivos", "exposição", "insalubridade",
               "periculosidade", "aposentadoria especial", "ltcat", "laudo técnico", "laudo", "perfil", "profissional"],
        "carta_concessao": ["carta de concessão", "benefício concedido", "rmi", "dib", "dip",
                           "der", "espécie", "nb", "comunicação de decisão", "concessão de benefício",
                           "inss concede", "benefício deferido", "benefício autorizado"],
        "guia_recolhimento": ["gps", "guia da previdência", "contribuição", "recolhimento", 
                             "competência", "autônomo", "facultativo", "guia de pagamento"]
    }
    
    # Conta ocorrências de palavras-chave
    counts = {doc_type: 0 for doc_type in keywords}
    text_lower = text.lower()
    
    for doc_type, terms in keywords.items():
        for term in terms:
            counts[doc_type] += text_lower.count(term)
    
    # Define as descrições dos documentos aqui para garantir que estejam sempre disponíveis
    doc_types = list(keywords.keys())
    doc_descriptions = [
        "Petição inicial com pedidos e fatos",
        "Contestação com argumentos de defesa",
        "Recurso com pedido de reforma da decisão",
        "Sentença judicial com decisão",
        "Despacho com determinações processuais",
        "Acórdão com decisão colegiada",
        "Carteira de Trabalho com registros de empregos",
        "Holerite com valores de salário e descontos",
        "Contrato de trabalho com condições laborais",
        "CNIS com histórico de contribuições previdenciárias",
        "PPP com informações sobre exposição a agentes nocivos",
        "Carta de concessão com dados do benefício previdenciário",
        "Guia de recolhimento de contribuição previdenciária"
    ]
    
        # Verifica se há termos específicos de carta de concessão com maior peso
    carta_concessao_termos_fortes = ["carta de concessão", "comunicação de decisão", "benefício concedido", "nb", "dib", "rmi", "espécie"]
    for termo in carta_concessao_termos_fortes:
        if termo in text_lower:
            counts["carta_concessao"] += 5  # Dá peso extra para estes termos
            
    # Calcula embeddings se o modelo estiver disponível
    model = get_sbert_model()
    if model:
        try:
            text_embedding = model.encode([text_lower[:1000]])[0]  # Usa apenas o início do texto
            desc_embeddings = model.encode(doc_descriptions)
            
            # Calcula similaridade
            similarities = cosine_similarity([text_embedding], desc_embeddings)[0]
            
            # Combina contagem de palavras-chave com similaridade semântica
            combined_scores = {
                doc_type: (counts[doc_type] * 0.7) + (similarities[i] * 0.3)
                for i, doc_type in enumerate(doc_types) if i < len(similarities)
            }
            
            # Determina o tipo mais provável
            most_likely_type = max(combined_scores, key=combined_scores.get)
            confidence = combined_scores[most_likely_type]
            
            logging.info(f"[DOCUMENT TYPE] Detectado: {most_likely_type} (confiança: {confidence:.2f})")
            
            return most_likely_type
        except Exception as e:
            logging.error(f"Erro ao calcular embeddings: {e}")
            # Fallback para contagem simples
    
    # Fallback: usa apenas contagem de palavras-chave
    most_likely_type = max(counts, key=counts.get)
    
    # Verifica se a contagem é muito baixa (documento não identificado claramente)
    if counts[most_likely_type] < 2:
        # Verifica se há padrões específicos de documentos previdenciários
        if any(term in text_lower for term in ["benefício", "inss", "previdenciário", "aposentadoria"]):
            if any(term in text_lower for term in ["concessão", "concedido", "deferido"]):
                return "carta_concessao"
    
    logging.info(f"[DOCUMENT TYPE] Detectado (contagem simples): {most_likely_type} (contagem: {counts[most_likely_type]})")
    return most_likely_type

def expand_question_for_legal_context(question: str) -> str:
    synonyms = {
        "autor": ["reclamante", "parte autora", "requerente", "demandante", "segurado", "trabalhador", "empregado"],
        "réu": ["reclamada", "empresa", "demandado", "parte ré", "empregador", "inss"],
        "advogado": ["procurador", "representante legal", "oab", "defensor", "advogado da parte"],
        "salário": ["remuneração", "vencimentos", "proventos", "contracheque", "holerite"],
        "benefício": ["aposentadoria", "auxílio", "pensão", "rmi", "renda mensal"],
        "contribuição": ["recolhimento", "tempo de serviço", "carência", "vínculo", "cnis"],
        "jornada": ["horário de trabalho", "horas extras", "banco de horas", "escala", "turno"],
        "verbas": ["rescisórias", "fgts", "férias", "13º", "aviso prévio", "multa"]
    }

    generic_terms = ["nome", "quem é", "qual o nome", "identificação"]

    expanded = question.lower()

    if any(term in expanded for term in generic_terms):
        # Adicionar todos os termos relacionados a partes do processo
        expanded += " autor reclamante réu reclamada parte segurado inss"

        # Adicionar termos específicos baseados no contexto da pergunta
    if "tempo" in expanded or "contribuição" in expanded:
        expanded += " cnis vínculos carência período especial insalubre"
    
    if "benefício" in expanded or "aposentadoria" in expanded:
        expanded += " rmi dib der dip nb espécie concessão"
    
    if "salário" in expanded or "pagamento" in expanded:
        expanded += " holerite contracheque remuneração verbas"
    
    if "jornada" in expanded or "hora" in expanded:
        expanded += " trabalho extra adicional noturno intervalo"
    
    if "argumento" in expanded or "contestação" in expanded or "recurso" in expanded:
        expanded += " defesa tese fundamento jurídico preliminar mérito"


    for key, terms in synonyms.items():
        if key in expanded:
            expanded += " " + " ".join(terms)
    return expanded

def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ASCII', 'ignore').decode('utf-8')
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

def extract_arguments_from_document(text: str, document_type: str) -> dict:
    """Extrai argumentos específicos de contestações e recursos para facilitar a elaboração de defesas"""
    arguments = {
        "preliminares": [],
        "merito": [],
        "pedidos": [],
        "fundamentos_legais": [],
        "jurisprudencia": []
    }
    
    # Detecta preliminares comuns
    preliminares_patterns = [
        r"preliminarmente[,\s]+([\w\s,.;]+?)(?=\n|No mérito|Quanto ao mérito)",
        r"das preliminares([\w\s,.;:]+?)(?=\n|No mérito|Quanto ao mérito)",
        r"prescrição([\w\s,.;:]+?)(?=\n|No mérito|Quanto ao mérito)"
    ]
    
    for pattern in preliminares_patterns:
        if match := re.search(pattern, text, re.IGNORECASE | re.DOTALL):
            preliminar = match.group(1).strip()
            arguments["preliminares"].append(preliminar)
    
    # Detecta argumentos de mérito
    if document_type == "contestação":
        merito_patterns = [
            r"(?:no mérito|quanto ao mérito)([\w\s,.;:]+?)(?=\n|Dos pedidos|Requer|Termos em que)",
            r"improcedência([\w\s,.;:]+?)(?=\n|Dos pedidos|Requer|Termos em que)"
        ]
    else:  # recurso
        merito_patterns = [
            r"(?:das razões recursais|razões de reforma)([\w\s,.;:]+?)(?=\n|Do pedido|Requer|Termos em que)",
            r"(?:merece reforma|deve ser reformada)([\w\s,.;:]+?)(?=\n|Do pedido|Requer|Termos em que)"
        ]
    
    for pattern in merito_patterns:
        if match := re.search(pattern, text, re.IGNORECASE | re.DOTALL):
            merito = match.group(1).strip()
            # Divide em parágrafos para separar argumentos
            paragrafos = re.split(r'\n+', merito)
            for paragrafo in paragrafos:
                if len(paragrafo.strip()) > 30:  # Ignora parágrafos muito curtos
                    arguments["merito"].append(paragrafo.strip())
    
    # Extrai pedidos
    pedidos_patterns = [
        r"(?:dos pedidos|requer|ante o exposto)([\w\s,.;:]+?)(?=\n|Termos em que|Nestes termos|Pede deferimento)",
    ]
    
    for pattern in pedidos_patterns:
        if match := re.search(pattern, text, re.IGNORECASE | re.DOTALL):
            pedido_text = match.group(1).strip()
            # Tenta identificar pedidos numerados ou com marcadores
            pedidos_numerados = re.findall(r'(?:\d+[\.|\)]\s*|\-\s*)([\w\s,.;:]+?)(?=\n|\d+[\.|\)]|\-|$)', pedido_text)
            if pedidos_numerados:
                arguments["pedidos"].extend([p.strip() for p in pedidos_numerados if len(p.strip()) > 10])
            else:
                arguments["pedidos"].append(pedido_text)
    
    # Extrai fundamentos legais
    fundamentos_patterns = [
        r'art(?:igo)?\.?\s*(\d+)[^\n\d]+(da|do)\s+([\w\s]+)',
        r'súmula\s+(\d+)[^\n\d]+(da|do)\s+([\w\s]+)',
        r'lei\s+(?:n[º°]?\s*)?(\d[\d\./]+)',
        r'decreto\s+(?:n[º°]?\s*)?(\d[\d\./]+)',
        r'CLT',
        r'Constituição Federal',
        r'CF/88'
    ]
    
    for pattern in fundamentos_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            fundamento = match.group(0)
            if fundamento not in arguments["fundamentos_legais"]:
                arguments["fundamentos_legais"].append(fundamento)
    
    # Extrai jurisprudência
    jurisprudencia_patterns = [
        r'(?:conforme|segundo)\s+(?:decidido|entendimento|jurisprudência)[^\n]+([\w\s,.;:]+?)(?=\n)',
        r'(?:TRT|TST|STF|STJ)[^\n]+([\w\s,.;:]+?)(?=\n)',
        r'(?:Tribunal|Corte)[^\n]+([\w\s,.;:]+?)(?=\n)'
    ]
    
    for pattern in jurisprudencia_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            jurisprudencia = match.group(0)
            if jurisprudencia not in arguments["jurisprudencia"]:
                arguments["jurisprudencia"].append(jurisprudencia)
    
    return arguments

def analyze_social_security_document(text: str, document_type: str) -> dict:
    """Analisa documentos previdenciários para extrair informações relevantes"""
    analysis = {
        "periodos_contribuicao": [],
        "tempo_total": "",
        "carencia": "",
        "atividades_especiais": [],
        "beneficios_anteriores": [],
        "rmi_calculada": "",
        "observacoes": []
    }
    
    # Extrai períodos de contribuição com padrão mais flexível
    periodos = re.findall(r'(\d{2}[/.-]\d{2}[/.-]\d{4})\s*(?:a|até|ao?|-)\s*(\d{2}[/.-]\d{2}[/.-]\d{4})', text)
    
    if periodos:
        for inicio, fim in periodos:
            # Normaliza o formato das datas
            inicio = inicio.replace('.', '/').replace('-', '/')
            fim = fim.replace('.', '/').replace('-', '/')
            
            # Busca contexto para encontrar o empregador
            pos_inicio = text.find(inicio)
            pos_fim = text.find(fim)
            if pos_inicio >= 0 and pos_fim >= 0:
                contexto = text[max(0, pos_inicio-100):min(len(text), pos_fim+100)]
                
                # Tenta encontrar o nome da empresa
                empresa = ""
                linhas_contexto = contexto.split('\n')
                for linha in linhas_contexto:
                    # Ignora linhas que contêm as datas para evitar falsos positivos
                    if inicio not in linha and fim not in linha:
                        # Busca por padrões comuns de nomes de empresas
                        if re.search(r'(ltda|s[/.]a|mei|eireli|empresa|empregador)', linha, re.IGNORECASE):
                            empresa = linha.strip()
                            break
                
                analysis["periodos_contribuicao"].append({
                    "inicio": inicio,
                    "fim": fim,
                    "empresa": empresa[:100] if empresa else ""  # Limita o tamanho
                })
        
        # Calcula tempo total se houver períodos
        if analysis["periodos_contribuicao"]:
            try:
                total_dias = 0
                for periodo in analysis["periodos_contribuicao"]:
                    inicio = datetime.strptime(periodo["inicio"], "%d/%m/%Y")
                    fim = datetime.strptime(periodo["fim"], "%d/%m/%Y")
                    dias = (fim - inicio).days
                    if dias > 0:  # Ignora períodos negativos (possíveis erros)
                        total_dias += dias
                
                anos = total_dias // 365
                meses = (total_dias % 365) // 30
                dias_restantes = (total_dias % 365) % 30
                
                analysis["tempo_total"] = f"{anos} anos, {meses} meses e {dias_restantes} dias"
                analysis["carencia"] = f"{len(analysis['periodos_contribuicao'])} competências"
            except Exception as e:
                logging.error(f"Erro ao calcular tempo total: {e}")
    
    # Busca por informações de benefício com padrão mais flexível
    nb_matches = re.findall(r'(?:Benefício|NB)[:\s]+(\d{10})', text, re.IGNORECASE)
    
    for nb in nb_matches:
        beneficio = {"nb": nb}
        
        # Busca contexto ao redor do NB
        pos_nb = text.find(nb)
        if pos_nb >= 0:
            contexto = text[max(0, pos_nb-200):min(len(text), pos_nb+500)]
            
            # Busca DIB, DIP, RMI próximos ao NB
            if dib_match := re.search(r'DIB[:\s]+(\d{2}/\d{2}/\d{4})', contexto, re.IGNORECASE):
                beneficio["dib"] = dib_match.group(1)
                
            if dip_match := re.search(r'DIP[:\s]+(\d{2}/\d{2}/\d{4})', contexto, re.IGNORECASE):
                beneficio["dip"] = dip_match.group(1)
                
            if rmi_match := re.search(r'RMI[:\s]+R\$\s*([\d.,]+)', contexto, re.IGNORECASE):
                beneficio["rmi"] = rmi_match.group(1)
                
            if especie_match := re.search(r'Espécie[:\s]+(\d{2})', contexto, re.IGNORECASE):
                beneficio["especie"] = especie_match.group(1)
        
        analysis["beneficios_anteriores"].append(beneficio)
    
    # Busca por atividades especiais/insalubres
    if document_type == "ppp" or "ppp" in text.lower() or "perfil profissiográfico" in text.lower():
        agentes_nocivos = re.findall(r'(?:Agente Nocivo|Agentes Nocivos|Fator de Risco)[:\s]+([\w\s,.;]+?)(?=\n|$)', text, re.IGNORECASE)
        for agente in agentes_nocivos:
            if len(agente.strip()) > 3:  # Ignora resultados muito curtos
                analysis["atividades_especiais"].append(agente.strip())
        
        # Busca por informações sobre EPI
        epi_info = re.search(r'EPI[:\s]+([\w\s,.;]+?)(?=\n|$)', text, re.IGNORECASE)
        if epi_info:
            analysis["observacoes"].append(f"EPI: {epi_info.group(1).strip()}")
    
    # Se não encontrou nenhuma informação específica, tenta extrair qualquer valor monetário
    if not analysis["rmi_calculada"] and not analysis["beneficios_anteriores"]:
        valores = re.findall(r'R\$\s*([\d.,]+)', text)
        if valores:
            # Filtra valores muito pequenos ou muito grandes
            valores_filtrados = [v for v in valores if 100 <= float(v.replace('.', '').replace(',', '.')) <= 50000]
            if valores_filtrados:
                analysis["observacoes"].append(f"Valores monetários encontrados: R$ {', R$ '.join(valores_filtrados[:5])}")
    
    return analysis

def analyze_labor_document(text: str, document_type: str) -> dict:
    """Analisa documentos trabalhistas para extrair informações relevantes"""
    analysis = {
        "contrato": {},
        "remuneracao": {},
        "jornada": {},
        "verbas_rescisoria": {},
        "observacoes": []
    }
    
    # Extrai informações de contrato de trabalho
    if document_type in ["contrato_trabalho", "ctps"]:
        # Dados básicos do contrato
        if match := re.search(r'Data de Admissão[:\s]+(\d{2}/\d{2}/\d{4})', text, re.IGNORECASE):
            analysis["contrato"]["data_admissao"] = match.group(1)
        if match := re.search(r'Data de Demissão[:\s]+(\d{2}/\d{2}/\d{4})', text, re.IGNORECASE):
            analysis["contrato"]["data_demissao"] = match.group(1)
        if match := re.search(r'Cargo[:/\s]+([\w\s]+?)(?=\n)', text, re.IGNORECASE):
            analysis["contrato"]["cargo"] = match.group(1).strip()
        if match := re.search(r'Função[:/\s]+([\w\s]+?)(?=\n)', text, re.IGNORECASE):
            analysis["contrato"]["funcao"] = match.group(1).strip()
        
        # Modalidade de contrato
        if re.search(r'prazo determinado', text, re.IGNORECASE):
            analysis["contrato"]["modalidade"] = "Prazo determinado"
        elif re.search(r'prazo indeterminado', text, re.IGNORECASE):
            analysis["contrato"]["modalidade"] = "Prazo indeterminado"
        elif re.search(r'experiência', text, re.IGNORECASE):
            analysis["contrato"]["modalidade"] = "Experiência"
        # Jornada de trabalho
        jornada_match = re.search(r'(?:Jornada|Horário)[:/\s]+([\w\s:àh]+?)(?=\n)', text, re.IGNORECASE)
        if jornada_match:
            analysis["jornada"]["descricao"] = jornada_match.group(1).strip()
            
            # Tenta extrair horários específicos
            horarios = re.findall(r'(\d{1,2})[h:.](\d{2})\s*(?:às|a|até|-)\s*(\d{1,2})[h:.](\d{2})', jornada_match.group(1))
            if horarios:
                entrada_hora, entrada_min, saida_hora, saida_min = horarios[0]
                analysis["jornada"]["entrada"] = f"{entrada_hora}:{entrada_min}"
                analysis["jornada"]["saida"] = f"{saida_hora}:{saida_min}"
                
                # Calcula horas diárias
                horas_diarias = int(saida_hora) - int(entrada_hora)
                if int(saida_min) < int(entrada_min):
                    horas_diarias -= 1
                analysis["jornada"]["horas_diarias"] = str(horas_diarias)
    
    # Extrai informações de holerites
    if document_type == "holerite":
        # Período de referência
        if match := re.search(r'(?:Período|Competência|Referente a)[:/\s]+(\d{2}/\d{4})', text, re.IGNORECASE):
            analysis["remuneracao"]["competencia"] = match.group(1)
        elif match := re.search(r'(?:Período|Competência|Referente a)[:/\s]+(\w+/\d{4})', text, re.IGNORECASE):
            analysis["remuneracao"]["competencia"] = match.group(1)
        
        # Salário base
        if match := re.search(r'(?:Salário Base|Salário|Vencimentos)[:/\s]+R\$\s*([\d.,]+)', text, re.IGNORECASE):
            analysis["remuneracao"]["salario_base"] = match.group(1)
        
        # Horas extras
        horas_extras = re.findall(r'(?:Hora Extra|HE|H\.E\.|Adicional de Hora)[^R]*R\$\s*([\d.,]+)', text, re.IGNORECASE)
        if horas_extras:
            analysis["remuneracao"]["horas_extras"] = sum([float(valor.replace('.', '').replace(',', '.')) for valor in horas_extras])
        
        # Adicionais
        if match := re.search(r'(?:Adicional Noturno|Ad\.Noturno)[^R]*R\$\s*([\d.,]+)', text, re.IGNORECASE):
            analysis["remuneracao"]["adicional_noturno"] = match.group(1)
        if match := re.search(r'(?:Adicional de Insalubridade|Insalubridade)[^R]*R\$\s*([\d.,]+)', text, re.IGNORECASE):
            analysis["remuneracao"]["adicional_insalubridade"] = match.group(1)
        if match := re.search(r'(?:Adicional de Periculosidade|Periculosidade)[^R]*R\$\s*([\d.,]+)', text, re.IGNORECASE):
            analysis["remuneracao"]["adicional_periculosidade"] = match.group(1)
        
        # Descontos
        if match := re.search(r'(?:INSS)[^R]*R\$\s*([\d.,]+)', text, re.IGNORECASE):
            analysis["remuneracao"]["desconto_inss"] = match.group(1)
        if match := re.search(r'(?:IRRF|Imposto de Renda)[^R]*R\$\s*([\d.,]+)', text, re.IGNORECASE):
            analysis["remuneracao"]["desconto_irrf"] = match.group(1)
        
        # Total
        if match := re.search(r'(?:Total Líquido|Líquido a Receber|Valor Líquido)[^R]*R\$\s*([\d.,]+)', text, re.IGNORECASE):
            analysis["remuneracao"]["liquido"] = match.group(1)
    
    # Extrai informações de termos de rescisão
    if re.search(r'(?:Termo de Rescisão|TRCT|Rescisão do Contrato)', text, re.IGNORECASE):
        # Motivo da rescisão
        if match := re.search(r'(?:Motivo|Causa)[:/\s]+([\w\s]+?)(?=\n)', text, re.IGNORECASE):
            analysis["verbas_rescisoria"]["motivo"] = match.group(1).strip()
        
        # Verbas rescisórias comuns
        verbas = {
            "saldo_salario": r'(?:Saldo de Salário)[^R]*R\$\s*([\d.,]+)',
            "aviso_previo": r'(?:Aviso Prévio)[^R]*R\$\s*([\d.,]+)',
            "ferias_proporcionais": r'(?:Férias Proporcionais|Férias Prop\.)[^R]*R\$\s*([\d.,]+)',
            "ferias_vencidas": r'(?:Férias Vencidas)[^R]*R\$\s*([\d.,]+)',
            "decimo_terceiro": r'(?:13º Salário|13º Prop\.)[^R]*R\$\s*([\d.,]+)',
            "fgts_rescisao": r'(?:FGTS Rescisão|Multa FGTS)[^R]*R\$\s*([\d.,]+)'
        }
        
        for key, pattern in verbas.items():
            if match := re.search(pattern, text, re.IGNORECASE):
                analysis["verbas_rescisoria"][key] = match.group(1)
        
        # Total da rescisão
        if match := re.search(r'(?:Total|Líquido)[^R]*R\$\s*([\d.,]+)', text, re.IGNORECASE):
            analysis["verbas_rescisoria"]["total"] = match.group(1)
    
    return analysis

def process_pdf_with_hybrid_extraction(pdf_bytes: bytes) -> str:
    """Processa PDF usando extração híbrida (PyPDF2 + OCR quando necessário)"""
    pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    full_text = ""
    extraction_methods = []
    
    for i, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text() or ""
        
        # Verifica se o texto extraído é suficiente
        if not page_text.strip() or len(page_text.strip()) < 100:
            logging.info(f"[OCR] Aplicando OCR na página {i+1} devido a texto insuficiente")
            try:
                # Converte apenas a página atual para imagem
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
    """Reordena documentos com base na similaridade semântica com a pergunta"""
        # Fallback se não houver documentos
    if not documents:
        return []
    # Expandir a pergunta com termos relacionados ao contexto jurídico trabalhista e previdenciário
    expanded_question = expand_question_for_legal_context(question)
    
    model = get_sbert_model()
    if not model:
        logging.warning("⚠️ SBERT indisponível - ignorando reranking semântico.")
        return documents[:5]
    
    # Adicionar contexto específico à pergunta
    if "réu" in question.lower() or "reu" in question.lower() or "reclamada" in question.lower():
        expanded_question += " reclamado demandado parte contrária empresa empregador"
    if "autor" in question.lower() or "reclamante" in question.lower():
        expanded_question += " reclamante requerente parte autora trabalhador empregado segurado"
    if "advogado" in question.lower():
        expanded_question += " procurador representante legal oab"
    if "salário" in question.lower() or "remuneração" in question.lower():
        expanded_question += " holerite contracheque pagamento vencimentos proventos"
    if "benefício" in question.lower() or "aposentadoria" in question.lower():
        expanded_question += " previdenciário inss rmi dib dip der carência"
    if "jornada" in question.lower() or "hora" in question.lower():
        expanded_question += " trabalho expediente turno escala intervalo descanso"
    if "argumento" in question.lower() or "contestação" in question.lower() or "recurso" in question.lower():
        expanded_question += " defesa tese fundamento jurídico preliminar mérito"
    
    # Calcular embeddings
    doc_texts = [doc.page_content for doc in documents]
    model = get_sbert_model()
    if not model:
        return documents
    doc_embeddings = model.encode(doc_texts)
    question_embedding = model.encode([expanded_question])[0]
    
    # Calcular similaridade
    scores = cosine_similarity([question_embedding], doc_embeddings)[0]
    
    # Ordenar documentos por similaridade
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    if max(scores) < 0.4:
        return []
    
    # Logging para depuração
    logging.info(f"[RERANKING] Pergunta expandida: {expanded_question}")
    for i, (doc, score) in enumerate(ranked[:3]):
        logging.info(f"[RERANKING] Doc {i+1}, Score: {score:.4f}, Preview: {doc.page_content[:80]}...")
        
    return [doc for doc, _ in ranked]

def extract_explicit_metadata(text: str) -> dict:
    metadata = {
        'fato': [],
        'prazo': [],
        'tipo': []
    }
    
    # Armazena uma amostra do texto para uso posterior
    metadata["text_sample"] = text[:5000]

    # ----------- INFORMAÇÕES BÁSICAS -----------
    # Extrai o tipo de ação - busca por padrões como "AÇÃO DE..."
    action_patterns = [
        r'(?:AÇÃO|Ação)\s+DE\s+([A-ZÀ-Ú\s]+)(?:COM|EM|cc)',
        r'propor\s+a\s+presente\s*\n+([A-ZÀ-Ú\s]+)(?:COM|EM|cc)',
        r'propor\s+a\s+presente\s+([A-ZÀ-Ú\s]+)(?:COM|EM|cc)'
    ]
    
    for pattern in action_patterns:
        if match := re.search(pattern, text, re.IGNORECASE):
            metadata["tipo_acao"] = match.group(1).strip()
            break
    
    # Extrai número do processo com formato mais flexível
    process_patterns = [
        r'(?:processo|autos)\s*(?:n[º°.]?)?:?\s*(\d{7}[-.]?\d{2}[-.]?\d{4}[-.]?\d[-.]?\d{4})',
        r'informe\s+o\s+processo\s+(\d{7}[-.]?\d{2}[-.]?\d{4}[-.]?\d[-.]?\d{4})',
        r'sob\s+o\s+número\s+(\d{7}[-.]?\d{2}[-.]?\d{4}[-.]?\d[-.]?\d{4})'
    ]
    
    for pattern in process_patterns:
        if match := re.search(pattern, text, re.IGNORECASE):
            metadata["numero_processo"] = match.group(1).strip()
            break
    
    # Extrai valor da causa com formato mais flexível
    value_match = re.search(r'[Vv]alor\s+da\s+causa\s*:?\s*R\$\s*([\d.,]+)', text)
    if value_match:
        metadata["valor_causa"] = value_match.group(1).strip()

    # ----------- EXTRAÇÃO DE PARTES E ADVOGADOS -----------
    # Busca por padrões de autor/requerente mais específicos
    author_patterns = [
        r'([A-ZÀ-Ú][A-ZÀ-Úa-zà-ú\s]+(?:LTDA|ME|EPP|S[/.]A[.]?|EIRELI)?)\s*(?:,|\.)\s*(?:pessoa jurídica|pessoa física)',
        r'([A-ZÀ-Ú][A-ZÀ-Úa-zà-ú\s]+(?:LTDA|ME|EPP|S[/.]A[.]?|EIRELI)?)\s*(?:,|\.)\s*(?:inscrita|inscrito)',
        r'([A-ZÀ-Ú][A-ZÀ-Úa-zà-ú\s]+(?:LTDA|ME|EPP|S[/.]A[.]?|EIRELI)?)\s*(?:,|\.)\s*estabelecida',
        r'([A-ZÀ-Ú][A-ZÀ-Úa-zà-ú\s]+(?:LTDA|ME|EPP|S[/.]A[.]?|EIRELI)?)\s*(?:,|\.)\s*(?:vem|por)',
        r'([A-ZÀ-Ú][A-ZÀ-Úa-zà-ú\s]+(?:LTDA|ME|EPP|S[/.]A[.]?|EIRELI)?)\s*(?:,|\.)\s*neste ato'
    ]
    
    for pattern in author_patterns:
        if match := re.search(pattern, text, re.IGNORECASE):
            author_name = match.group(1).strip()
            if len(author_name) > 5 and not any(term in author_name.upper() for term in ["EXCELENTÍSSIMO", "JUIZ", "DOUTOR"]):
                metadata["autor"] = author_name
                break
    
    # Busca por padrões de réu/requerido mais específicos
    defendant_patterns = [
        r'[Ee]m\s+face\s+(?:do|da)\s+([A-ZÀ-Ú][A-ZÀ-Úa-zà-ú\s]+(?:LTDA|ME|EPP|S[/.]A[.]?|EIRELI)?)',
        r'[Cc]ontra\s+(?:o|a)\s+([A-ZÀ-Ú][A-ZÀ-Úa-zà-ú\s]+(?:LTDA|ME|EPP|S[/.]A[.]?|EIRELI)?)',
        r'[Cc]itar\s+(?:o|a)\s+([A-ZÀ-Ú][A-ZÀ-Úa-zà-ú\s]+(?:LTDA|ME|EPP|S[/.]A[.]?|EIRELI)?)'
    ]
    
    for pattern in defendant_patterns:
        if match := re.search(pattern, text, re.IGNORECASE):
            defendant_name = match.group(1).strip()
            # Limpa o nome do réu, removendo tudo após a primeira vírgula ou ponto
            if "," in defendant_name:
                defendant_name = defendant_name.split(",")[0].strip()
            if "." in defendant_name:
                defendant_name = defendant_name.split(".")[0].strip()
            if "cadastrada" in defendant_name:
                defendant_name = defendant_name.split("cadastrada")[0].strip()
            
            if len(defendant_name) > 5:
                metadata["reu"] = defendant_name
                break
    
    # Extrai CPF/CNPJ das partes com padrões mais específicos
    cnpj_patterns = [
        r'(?:CNPJ|CGC)\s*(?:Sob|sob)?\s*(?:N[º°.]?|n[º°.]?)?\s*([\d]{2}\.[\d]{3}\.[\d]{3}/[\d]{4}-[\d]{2})',
        r'(?:CNPJ|CGC)\s*(?:Sob|sob)?\s*(?:N[º°.]?|n[º°.]?)?\s*([\d]{2}\.[\d]{3}\.[\d]{3}/[\d]{4})',
        r'(?:CNPJ|CGC)\s*(?:Sob|sob)?\s*(?:N[º°.]?|n[º°.]?)?\s*([\d]{14})',
        r'(?:CNPJ|CGC)\s*(?:Sob|sob)?\s*(?:N[º°.]?|n[º°.]?)?\s*([\d]{2}[\.]?[\d]{3}[\.]?[\d]{3}[/]?[\d]{4}[-]?[\d]{2})',
        r'inscrita\s+no\s+(?:CNPJ|CGC)\s*(?:Sob|sob)?\s*(?:N[º°.]?|n[º°.]?)?\s*([\d]{2}[\.]?[\d]{3}[\.]?[\d]{3}[/]?[\d]{4}[-]?[\d]{2})'
    ]
    
    cpf_patterns = [
        r'(?:CPF|RG)\s*(?:Sob|sob)?\s*(?:N[º°.]?|n[º°.]?)?\s*([\d]{3}\.[\d]{3}\.[\d]{3}-[\d]{2})',
        r'(?:CPF|RG)\s*(?:Sob|sob)?\s*(?:N[º°.]?|n[º°.]?)?\s*([\d]{3}\.[\d]{3}\.[\d]{3})',
        r'(?:CPF|RG)\s*(?:Sob|sob)?\s*(?:N[º°.]?|n[º°.]?)?\s*([\d]{11})',
        r'(?:CPF|RG)\s*(?:Sob|sob)?\s*(?:N[º°.]?|n[º°.]?)?\s*([\d]{3}[\.]?[\d]{3}[\.]?[\d]{3}[-]?[\d]{2})'
    ]
    
    # Busca CNPJ do autor
    for pattern in cnpj_patterns:
        if match := re.search(pattern, text[:text.find("Em face") if "Em face" in text else len(text)], re.IGNORECASE):
            metadata["cpf_cnpj_autor"] = match.group(1).strip()
            break
    
    # Busca CPF do autor se não encontrou CNPJ
    if "cpf_cnpj_autor" not in metadata:
        for pattern in cpf_patterns:
            if match := re.search(pattern, text[:text.find("Em face") if "Em face" in text else len(text)], re.IGNORECASE):
                metadata["cpf_cnpj_autor"] = match.group(1).strip()
                break
    
    # Busca CNPJ do réu
    for pattern in cnpj_patterns:
        if match := re.search(pattern, text[text.find("Em face") if "Em face" in text else 0:], re.IGNORECASE):
            metadata["cpf_cnpj_reu"] = match.group(1).strip()
            break
    
    # Busca CPF do réu se não encontrou CNPJ
    if "cpf_cnpj_reu" not in metadata:
        for pattern in cpf_patterns:
            if match := re.search(pattern, text[text.find("Em face") if "Em face" in text else 0:], re.IGNORECASE):
                metadata["cpf_cnpj_reu"] = match.group(1).strip()
                break
    
    # Busca por advogado com padrões mais específicos
    lawyer_patterns = [
        r'por\s+seu\s+advogado\s+([A-ZÀ-Ú][A-ZÀ-Úa-zà-ú\s]+)',
        r'advogado\s+([A-ZÀ-Ú][A-ZÀ-Úa-zà-ú\s]+),\s*OAB',
        r'advogado\s+([A-ZÀ-Ú][A-ZÀ-Úa-zà-ú\s]+),\s*inscrito'
    ]
    
    for pattern in lawyer_patterns:
        if match := re.search(pattern, text, re.IGNORECASE):
            lawyer_name = match.group(1).strip()
            if len(lawyer_name) > 5 and not any(term in lawyer_name.upper() for term in ["EXCELENTÍSSIMO", "JUIZ", "DOUTOR"]):
                metadata["advogado"] = lawyer_name
                break
    
    # Busca por OAB com padrões mais específicos
    oab_patterns = [
        r'OAB[:/\s]*([A-Z]{2}[/\s]*\d+)',
        r'inscrito\s+na\s+OAB[:/\s]*([A-Z]{2}[/\s]*\d+)',
        r'advogado\s+inscrito\s+sob\s+n[º°.]?\s*([A-Z]{2}[/\s]*\d+)'
    ]
    
    for pattern in oab_patterns:
        if match := re.search(pattern, text, re.IGNORECASE):
            metadata["oab_advogado"] = match.group(1).strip()
            break
    
    # ----------- EXTRAÇÃO DE VALORES MONETÁRIOS -----------
    # Extrai todos os valores monetários mencionados com padrões mais específicos
    monetary_values = []
    
    # Padrão para valores com R$
    money_patterns = [
        r'R\$\s*([\d.]+,\d{2})',  # R$ 1.234,56
        r'R\$\s*([\d]+,\d{2})',   # R$ 1234,56
        r'R\$\s*([\d.,]+)',       # Outros formatos
    ]
    
    for pattern in money_patterns:
        for match in re.finditer(pattern, text):
            value = match.group(1).strip()
            # Normaliza o valor
            if ',' in value:
                # Formato brasileiro (R$ 1.234,56)
                value = value.replace('.', '').replace(',', '.')
            monetary_values.append(value)
    
    if monetary_values:
        # Remove duplicatas e limita a 5 valores
        unique_values = []
        for value in monetary_values:
            if value not in unique_values:
                unique_values.append(value)
                if len(unique_values) >= 5:
                    break
        
        metadata["valores_monetarios"] = [f"R$ {v}" for v in unique_values]
    
    # ----------- EXTRAÇÃO DE DATAS -----------
    # Extrai todas as datas mencionadas
    dates = re.findall(r'\d{2}/\d{2}/\d{4}', text)
    if dates:
        # Remove duplicatas e limita a 5 datas
        unique_dates = []
        for date in dates:
            if date not in unique_dates:
                unique_dates.append(date)
                if len(unique_dates) >= 5:
                    break
        
        metadata["datas_mencionadas"] = unique_dates
    
    def extract_petition_details(text: str) -> dict:
        """Extrai detalhes específicos de petições iniciais"""
        details = {}
        
        # Extrai os pedidos da petição
        pedidos_patterns = [
            r'(?:DOS PEDIDOS|PEDIDOS|DO PEDIDO)[:\s]+([\s\S]+?)(?=\n\s*\n|TERMOS EM QUE|NESTES TERMOS|PEDE DEFERIMENTO)',
            r'(?:requer|REQUER)[:\s]+([\s\S]+?)(?=\n\s*\n|TERMOS EM QUE|NESTES TERMOS|PEDE DEFERIMENTO)',
            r'(?:Ante o exposto|ANTE O EXPOSTO)[:\s]+([\s\S]+?)(?=\n\s*\n|TERMOS EM QUE|NESTES TERMOS|PEDE DEFERIMENTO)'
        ]
        
        for pattern in pedidos_patterns:
            if match := re.search(pattern, text, re.IGNORECASE):
                pedidos_text = match.group(1).strip()
                
                # Tenta identificar pedidos numerados
                pedidos_numerados = re.findall(r'(?:\d+[\.|\)]\s*|\-\s*|[a-z]\)\s*)([\w\s,.;:]+?)(?=\n|\d+[\.|\)]|\-|[a-z]\)|$)', pedidos_text)
                
                if pedidos_numerados:
                    details["pedidos"] = [p.strip() for p in pedidos_numerados if len(p.strip()) > 10]
                else:
                    # Se não encontrou pedidos numerados, tenta dividir por parágrafos
                    paragrafos = re.split(r'\n+', pedidos_text)
                    details["pedidos"] = [p.strip() for p in paragrafos if len(p.strip()) > 10]
                
                break
        
        # Extrai os fatos narrados
        fatos_patterns = [
            r'(?:DOS FATOS|FATOS|DOS FATOS E FUNDAMENTOS)[:\s]+([\s\S]+?)(?=\n\s*\n|DO DIREITO|DOS PEDIDOS|PEDIDOS|DO PEDIDO)',
            r'(?:BREVE RELATO DOS FATOS|SÍNTESE DOS FATOS)[:\s]+([\s\S]+?)(?=\n\s*\n|DO DIREITO|DOS PEDIDOS|PEDIDOS|DO PEDIDO)'
        ]
        
        for pattern in fatos_patterns:
            if match := re.search(pattern, text, re.IGNORECASE):
                fatos_text = match.group(1).strip()
                
                # Divide os fatos em parágrafos
                paragrafos = re.split(r'\n+', fatos_text)
                details["fatos"] = [p.strip() for p in paragrafos if len(p.strip()) > 10]
                
                break
        
        # Extrai os fundamentos jurídicos
        fundamentos_patterns = [
            r'(?:DO DIREITO|DIREITO|FUNDAMENTOS JURÍDICOS)[:\s]+([\s\S]+?)(?=\n\s*\n|DOS PEDIDOS|PEDIDOS|DO PEDIDO)',
            r'(?:FUNDAMENTOS|DA FUNDAMENTAÇÃO)[:\s]+([\s\S]+?)(?=\n\s*\n|DOS PEDIDOS|PEDIDOS|DO PEDIDO)'
        ]
        
        for pattern in fundamentos_patterns:
            if match := re.search(pattern, text, re.IGNORECASE):
                fundamentos_text = match.group(1).strip()
                
                # Divide os fundamentos em parágrafos
                paragrafos = re.split(r'\n+', fundamentos_text)
                details["fundamentos"] = [p.strip() for p in paragrafos if len(p.strip()) > 10]
                
                break
        
        # Extrai fundamentos legais citados (artigos, leis, etc.)
        fundamentos_legais = []
        
        legal_patterns = [
            r'(?:art(?:igo)?\.?\s*(\d+)[^\n\d]+(da|do)\s+([\w\s]+))',
            r'(?:Lei\s+(?:n[º°]?\s*)?(\d[\d\./]+))',
            r'(?:Código\s+([\w\s]+))',
            r'(?:Súmula\s+(\d+)[^\n\d]+(da|do)\s+([\w\s]+))',
            r'(?:CF|Constituição Federal)',
            r'(?:CLT|Consolidação das Leis do Trabalho)',
            r'(?:CPC|Código de Processo Civil)'
        ]
        
        for pattern in legal_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                fundamento = match.group(0).strip()
                if fundamento not in fundamentos_legais:
                    fundamentos_legais.append(fundamento)
        
        if fundamentos_legais:
            details["fundamentos_legais"] = fundamentos_legais
        
        # Extrai valor da causa
        valor_causa_patterns = [
            r'[Vv]alor\s+da\s+causa\s*:?\s*R\$\s*([\d.,]+)',
            r'[Dd]á-se\s+à\s+causa\s+o\s+valor\s+de\s*R\$\s*([\d.,]+)'
        ]
        
        for pattern in valor_causa_patterns:
            if match := re.search(pattern, text):
                details["valor_causa"] = match.group(1).strip()
                break
        
        # Extrai pedido de tutela/liminar
        tutela_patterns = [
            r'(?:TUTELA ANTECIPADA|TUTELA DE URGÊNCIA|LIMINAR)[:\s]+([\s\S]+?)(?=\n\s*\n|DOS PEDIDOS|PEDIDOS|DO PEDIDO)',
            r'(?:requer|REQUER)[^.]*(?:liminarmente|LIMINARMENTE)[:\s]+([\s\S]+?)(?=\n\s*\n|DOS PEDIDOS|PEDIDOS|DO PEDIDO)'
        ]
        
        for pattern in tutela_patterns:
            if match := re.search(pattern, text, re.IGNORECASE):
                tutela_text = match.group(1).strip()
                
                # Divide o pedido de tutela em parágrafos
                paragrafos = re.split(r'\n+', tutela_text)
                details["pedido_tutela"] = [p.strip() for p in paragrafos if len(p.strip()) > 10]
                
                break
        
        # Extrai provas mencionadas
        provas_patterns = [
            r'(?:DAS PROVAS|PROVAS|DA PROVA)[:\s]+([\s\S]+?)(?=\n\s*\n|DOS PEDIDOS|PEDIDOS|DO PEDIDO)',
            r'(?:provar[á]? o alegado por todos os meios)[:\s]+([\s\S]+?)(?=\n\s*\n|DOS PEDIDOS|PEDIDOS|DO PEDIDO)'
        ]
        
        for pattern in provas_patterns:
            if match := re.search(pattern, text, re.IGNORECASE):
                provas_text = match.group(1).strip()
                
                # Divide as provas em itens
                itens = re.findall(r'(?:\d+[\.|\)]\s*|\-\s*|[a-z]\)\s*)([\w\s,.;:]+?)(?=\n|\d+[\.|\)]|\-|[a-z]\)|$)', provas_text)
                
                if itens:
                    details["provas"] = [p.strip() for p in itens if len(p.strip()) > 5]
                else:
                    # Se não encontrou itens, tenta dividir por parágrafos
                    paragrafos = re.split(r'\n+', provas_text)
                    details["provas"] = [p.strip() for p in paragrafos if len(p.strip()) > 5]
                
                break
        
        return details

        # Verifica se o documento parece ser uma petição inicial
    if re.search(r'(?:AÇÃO|Ação)\s+DE|propor\s+a\s+presente', text, re.IGNORECASE):
        petition_details = extract_petition_details(text)
        if petition_details:
            metadata.update(petition_details)
            
    return metadata

def extract_judicial_decisions(text: str) -> list:
    """Extrai decisões judiciais do documento"""
    decisions = []
    
    # Padrões para identificar decisões judiciais
    decision_patterns = [
        r'(?:DECIDO|DECIDO:|DECISÃO|DECISÃO:)[^\n]*([\s\S]+?)(?=\n\s*\n|INTIMEM-SE|PUBLIQUE-SE|CUMPRA-SE|$)',
        r'(?:DEFIRO|INDEFIRO|DETERMINO|HOMOLOGO)[^\n]*([\s\S]+?)(?=\n\s*\n|INTIMEM-SE|PUBLIQUE-SE|CUMPRA-SE|$)',
        r'(?:Pelo exposto|Ante o exposto|Diante do exposto|Isto posto)[^\n]*([\s\S]+?)(?=\n\s*\n|INTIMEM-SE|PUBLIQUE-SE|CUMPRA-SE|$)'
    ]
    
    for pattern in decision_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            decision_text = match.group(1).strip()
            if len(decision_text) > 20:  # Ignora textos muito curtos
                # Tenta identificar o tipo de decisão
                decision_type = "Decisão"
                if re.search(r'sentença|julgo|procedente|improcedente', decision_text, re.IGNORECASE):
                    decision_type = "Sentença"
                elif re.search(r'liminar|tutela|antecipação|urgência', decision_text, re.IGNORECASE):
                    decision_type = "Decisão Liminar"
                elif re.search(r'designo|designe-se|marcar|agendar', decision_text, re.IGNORECASE):
                    decision_type = "Despacho de Agendamento"
                
                # Tenta extrair a data da decisão
                date_match = re.search(r'\d{2}/\d{2}/\d{4}', decision_text)
                decision_date = date_match.group(0) if date_match else "Data não identificada"
                
                decisions.append({
                    "tipo": decision_type,
                    "data": decision_date,
                    "conteudo": decision_text
                })
    
    return decisions

def extract_procedural_deadlines(text: str) -> list:
    """Extrai prazos processuais do documento"""
    deadlines = []
    
    # Padrões para identificar prazos
    deadline_patterns = [
        r'prazo\s+de\s+(\d+|cinco|dez|quinze|vinte|trinta)\s+dias',
        r'(\d+|cinco|dez|quinze|vinte|trinta)\s+dias\s+(?:úteis|corridos)?',
        r'até\s+(?:o\s+dia\s+)?(\d{2}/\d{2}/\d{4})',
        r'no\s+prazo\s+(?:legal|comum|de\s+(\d+|cinco|dez|quinze|vinte|trinta)\s+dias)'
    ]
    
    for pattern in deadline_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # Extrai o contexto da menção ao prazo (30 caracteres antes e depois)
            start_pos = max(0, match.start() - 30)
            end_pos = min(len(text), match.end() + 30)
            context = text[start_pos:end_pos]
            
            # Tenta identificar a finalidade do prazo
            purpose = "Não especificado"
            if re.search(r'contestar|contestação', context, re.IGNORECASE):
                purpose = "Contestação"
            elif re.search(r'recurso|recorrer|apelar|apelação', context, re.IGNORECASE):
                purpose = "Recurso"
            elif re.search(r'manifestar|manifestação', context, re.IGNORECASE):
                purpose = "Manifestação"
            elif re.search(r'cumprir|cumprimento', context, re.IGNORECASE):
                purpose = "Cumprimento"
            elif re.search(r'pagar|pagamento', context, re.IGNORECASE):
                purpose = "Pagamento"
            
            deadlines.append({
                "prazo": match.group(0),
                "finalidade": purpose,
                "contexto": context
            })
    
    return deadlines

def extract_hearings(text: str) -> list:
    """Extrai informações sobre audiências designadas"""
    hearings = []
    
    # Padrões para identificar audiências
    hearing_patterns = [
        r'audiência\s+(?:de\s+)?(conciliação|instrução|julgamento|inicial|una)\s+(?:para\s+)?(?:o\s+dia\s+)?(\d{2}/\d{2}/\d{4})(?:\s+às\s+(\d{1,2}[h:]\d{2}))?',
        r'designo\s+(?:audiência|perícia)\s+(?:para\s+)?(?:o\s+dia\s+)?(\d{2}/\d{2}/\d{4})(?:\s+às\s+(\d{1,2}[h:]\d{2}))?',
        r'(?:dia|data)\s+(\d{2}/\d{2}/\d{4})(?:\s+às\s+(\d{1,2}[h:]\d{2}))?\s+para\s+(?:audiência|perícia)'
    ]
    
    for pattern in hearing_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # Extrai o contexto da menção à audiência (50 caracteres antes e depois)
            start_pos = max(0, match.start() - 50)
            end_pos = min(len(text), match.end() + 50)
            context = text[start_pos:end_pos]
            
            # Tenta identificar o tipo de audiência
            hearing_type = "Não especificado"
            if re.search(r'conciliação', context, re.IGNORECASE):
                hearing_type = "Conciliação"
            elif re.search(r'instrução', context, re.IGNORECASE):
                hearing_type = "Instrução"
            elif re.search(r'julgamento', context, re.IGNORECASE):
                hearing_type = "Julgamento"
            elif re.search(r'inicial', context, re.IGNORECASE):
                hearing_type = "Inicial"
            elif re.search(r'una', context, re.IGNORECASE):
                hearing_type = "Una"
            elif re.search(r'perícia', context, re.IGNORECASE):
                hearing_type = "Perícia"
            
            # Extrai a data e hora
            date_match = re.search(r'\d{2}/\d{2}/\d{4}', context)
            time_match = re.search(r'\d{1,2}[h:]\d{2}', context)
            
            date = date_match.group(0) if date_match else "Data não identificada"
            time = time_match.group(0) if time_match else "Horário não identificado"
            
            hearings.append({
                "tipo": hearing_type,
                "data": date,
                "hora": time,
                "contexto": context
            })
    
    return hearings

def extract_controversial_points(text: str, document_type: str) -> dict:
    """Extrai pontos controvertidos de fato e direito"""
    controversial_points = {
        "fato": [],
        "direito": []
    }
    
    # Se for uma contestação, busca por negativas de fatos alegados pelo autor
    if document_type == "contestação":
        # Busca por negativas de fatos
        denial_patterns = [
            r'(?:nega|impugna|contesta|não procede|inverídico|inverdade)[^\n.]*(?:alegação|afirmação|fato)[^\n.]*(?:autor|reclamante)',
            r'(?:não é verdade|não ocorreu|jamais ocorreu|nunca aconteceu)[^\n.]*',
            r'(?:diferentemente|ao contrário|diversamente)[^\n.]*(?:do que alega|do alegado|do afirmado)[^\n.]*(?:autor|reclamante)'
        ]
        
        for pattern in denial_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Extrai o contexto da negativa (50 caracteres antes e depois)
                start_pos = max(0, match.start() - 50)
                end_pos = min(len(text), match.end() + 50)
                context = text[start_pos:end_pos]
                
                controversial_points["fato"].append(context.strip())
        
        # Busca por divergências de interpretação legal
        legal_disagreement_patterns = [
            r'(?:não se aplica|inaplicável|não incide)[^\n.]*(?:artigo|art\.|lei|código|súmula)',
            r'(?:interpretação|entendimento)[^\n.]*(?:equivocada|errônea|incorreta)',
            r'(?:jurisprudência|precedente|julgado)[^\n.]*(?:contrária|contrário|desfavorável)'
        ]
        
        for pattern in legal_disagreement_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Extrai o contexto da divergência legal (50 caracteres antes e depois)
                start_pos = max(0, match.start() - 50)
                end_pos = min(len(text), match.end() + 50)
                context = text[start_pos:end_pos]
                
                controversial_points["direito"].append(context.strip())
    
    # Se for uma réplica ou manifestação sobre contestação, busca por reafirmações do autor
    elif "réplica" in text.lower() or "manifestação" in text.lower():
        reaffirmation_patterns = [
            r'(?:reitera|reafirma|mantém|insiste)[^\n.]*(?:alegação|afirmação|fato|pedido)',
            r'(?:improcedente|não merece acolhimento)[^\n.]*(?:alegação|afirmação|argumento)[^\n.]*(?:réu|reclamada)',
            r'(?:diferentemente|ao contrário|diversamente)[^\n.]*(?:do que alega|do alegado|do afirmado)[^\n.]*(?:réu|reclamada)'
        ]
        
        for pattern in reaffirmation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Extrai o contexto da reafirmação (50 caracteres antes e depois)
                start_pos = max(0, match.start() - 50)
                end_pos = min(len(text), match.end() + 50)
                context = text[start_pos:end_pos]
                
                controversial_points["fato"].append(context.strip())
    
    # Busca por menções explícitas a pontos controvertidos
    explicit_patterns = [
        r'(?:ponto|matéria|questão)[^\n.]*(?:controvertida|controvertido|controversa|controverso)',
        r'(?:controvérsia|divergência)[^\n.]*(?:fato|direito|interpretação)',
        r'(?:fato|matéria)[^\n.]*(?:controvertido|controverso|em discussão|em debate)'
    ]
    
    for pattern in explicit_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # Extrai o contexto da menção explícita (50 caracteres antes e depois)
            start_pos = max(0, match.start() - 50)
            end_pos = min(len(text), match.end() + 50)
            context = text[start_pos:end_pos]
            
            # Tenta classificar se é ponto controvertido de fato ou de direito
            if re.search(r'fato|ocorr|acontec|realiz|execut', context, re.IGNORECASE):
                controversial_points["fato"].append(context.strip())
            elif re.search(r'direito|lei|artigo|código|jurídic|legal', context, re.IGNORECASE):
                controversial_points["direito"].append(context.strip())
            else:
                # Se não conseguir classificar, coloca em ambos
                controversial_points["fato"].append(context.strip())
                controversial_points["direito"].append(context.strip())
    
    # Remove duplicatas
    controversial_points["fato"] = list(set(controversial_points["fato"]))
    controversial_points["direito"] = list(set(controversial_points["direito"]))
    
    return controversial_points
            
def format_metadata_for_prompt(metadata: dict) -> str:
    if not metadata:
        return "Nenhum metadado detectado."
    
    formatted_lines = []
    for k, v in metadata.items():
        # Pula a chave text_sample que pode ser muito grande
        if k == "text_sample":
            continue
            
        # Verifica se o valor existe e não é vazio
        if v:
            # Formata dicionários aninhados
            if isinstance(v, dict):
                nested_lines = []
                for nk, nv in v.items():
                    if nv:  # Verifica se o valor aninhado não é vazio
                        nested_lines.append(f"  {nk.replace('_', ' ').capitalize()}: {nv}")
                if nested_lines:
                    formatted_lines.append(f"{k.replace('_', ' ').capitalize()}:")
                    formatted_lines.extend(nested_lines)
            else:
                formatted_lines.append(f"{k.replace('_', ' ').capitalize()}: {v}")
    
    if not formatted_lines:
        return "Nenhum metadado relevante detectado."
        
    return "\n".join(formatted_lines)

def ensure_metadata_keys(metadata):
    """Garante que todas as chaves necessárias existam no dicionário de metadados"""
    if metadata is None:
        metadata = {}
    
    # Lista de todas as chaves que podem ser acessadas em qualquer lugar do código
    required_keys = ['fato', 'prazo', 'tipo', 'preliminares', 'merito', 'pedidos', 
                    'fundamentos_legais', 'jurisprudencia', 'analise_especializada']
    
    # Garante que todas as chaves existam
    for key in required_keys:
        if key not in metadata:
            metadata[key] = []
    
    # Garante que analise_especializada exista e tenha todas as subchaves necessárias
    if 'analise_especializada' not in metadata:
        metadata['analise_especializada'] = {}
    
    if isinstance(metadata['analise_especializada'], dict):
        subkeys = ['periodos_contribuicao', 'tempo_total', 'carencia', 'atividades_especiais',
                  'beneficios_anteriores', 'rmi_calculada', 'observacoes', 'contrato',
                  'remuneracao', 'jornada', 'verbas_rescisoria']
        
        for key in subkeys:
            if key not in metadata['analise_especializada']:
                if key in ['contrato', 'remuneracao', 'jornada', 'verbas_rescisoria']:
                    metadata['analise_especializada'][key] = {}
                else:
                    metadata['analise_especializada'][key] = []
    
    return metadata

def build_adaptive_prompt(query: str, metadata: dict):
    """Cria um prompt adaptativo baseado no tipo de pergunta do usuário"""
    # Garantir que metadata existe e é um dicionário
    if metadata is None:
        metadata = {}
    
    # Aplicar a função ensure_metadata_keys para garantir que todas as chaves existam
    metadata = ensure_metadata_keys(metadata)
    
    # Formatar os metadados para o prompt
    metadata_str = format_metadata_for_prompt(metadata)
    
    # Detecta o tipo de pergunta
    query_lower = query.lower()
    
    # Instruções específicas baseadas no tipo de pergunta
    specific_instructions = ""
    
    if any(term in query_lower for term in ["razões de fato", "fatos", "fato", "aconteceu", "ocorreu"]):
        specific_instructions = """
        Ao responder sobre RAZÕES DE FATO:
        - Identifique e enumere cronologicamente os fatos narrados
        - Separe claramente os fatos alegados pelo autor dos fatos alegados pelo réu
        - Destaque contradições factuais entre as alegações das partes
        - Indique quais fatos são controversos e quais são incontroversos
        - Relacione os fatos com os documentos juntados que os comprovam
        - Organize sua resposta em tópicos numerados
        """
    elif any(term in query_lower for term in ["razões de direito", "direito", "fundamentos", "legislação", "jurisprudência"]):
        specific_instructions = """
        Ao responder sobre RAZÕES DE DIREITO:
        - Identifique os principais fundamentos jurídicos invocados
        - Liste a legislação citada (artigos, leis, decretos, etc.)
        - Mencione a jurisprudência citada (súmulas, julgados, etc.)
        - Destaque as teses jurídicas controversas entre as partes
        - Indique possíveis omissões ou fragilidades na fundamentação
        - Organize sua resposta em tópicos, separando por temas jurídicos
        """
    elif any(term in query_lower for term in ["pedidos", "requer", "tutela", "liminar", "antecipação"]):
        specific_instructions = """
        Ao responder sobre PEDIDOS:
        - Enumere todos os pedidos formulados na ordem em que aparecem
        - Separe pedidos principais de pedidos acessórios
        - Identifique pedidos de tutela antecipada/liminar
        - Relacione cada pedido com os fatos e fundamentos que o sustentam
        - Indique se há pedidos genéricos ou pedidos subsidiários
        - Apresente os pedidos em lista numerada
        """
    elif any(term in query_lower for term in ["documentos", "provas", "juntados", "anexos"]):
        specific_instructions = """
        Ao responder sobre DOCUMENTOS JUNTADOS:
        - Liste todos os documentos mencionados no texto
        - Classifique os documentos por tipo (contratos, recibos, laudos, etc.)
        - Indique a relevância de cada documento para os fatos alegados
        - Destaque documentos essenciais para a comprovação das alegações
        - Identifique possíveis documentos faltantes ou necessários
        - Organize em lista, agrupando por categoria
        """
    elif any(term in query_lower for term in ["pontos controvertidos", "controvérsia", "divergência"]):
        specific_instructions = """
        Ao responder sobre PONTOS CONTROVERTIDOS:
        - Identifique claramente os pontos de FATO controvertidos
        - Identifique claramente os pontos de DIREITO controvertidos
        - Relacione cada controvérsia com os argumentos de cada parte
        - Indique quais provas seriam necessárias para resolver cada controvérsia
        - Destaque as principais divergências de interpretação jurídica
        - Organize em tópicos separados para fato e direito
        """
    elif any(term in query_lower for term in ["intimações", "citação", "notificação", "prazo"]):
        specific_instructions = """
        Ao responder sobre INTIMAÇÕES e PRAZOS:
        - Verifique se as intimações foram realizadas corretamente
        - Identifique as datas de intimação/citação mencionadas
        - Calcule os prazos processuais relevantes
        - Indique se houve cumprimento tempestivo dos prazos
        - Destaque possíveis nulidades nas intimações
        - Apresente as informações em ordem cronológica
        """
    elif any(term in query_lower for term in ["decisão", "despacho", "sentença", "juiz", "magistrado"]):
        specific_instructions = """
        Ao responder sobre DECISÕES JUDICIAIS:
        - Identifique todas as decisões mencionadas no documento
        - Resuma o conteúdo de cada decisão de forma objetiva
        - Destaque os principais fundamentos utilizados pelo juiz
        - Indique quais pedidos foram deferidos ou indeferidos
        - Mencione eventuais determinações ou providências ordenadas
        - Organize cronologicamente, da mais antiga para a mais recente
        """
    elif any(term in query_lower for term in ["audiência", "perícia", "prova", "testemunha"]):
        specific_instructions = """
        Ao responder sobre AUDIÊNCIAS e PERÍCIAS:
        - Identifique as datas e horários designados
        - Indique o tipo/finalidade da audiência ou perícia
        - Mencione quais pessoas devem comparecer
        - Destaque eventuais determinações específicas do juiz
        - Informe sobre provas a serem produzidas
        - Apresente as informações em formato de agenda
        """

    system_template = f"""
    Você é um assistente jurídico especializado em direito trabalhista e previdenciário. Use as informações fornecidas no contexto para responder às perguntas do usuário sobre o documento enviado.

    Se a informação estiver presente no documento, forneça uma resposta direta, objetiva e estruturada.
    
⚠️ Se a informação **não estiver explicitamente presente**, siga esta diretriz:
    - Explique se há previsão, instrução ou citação indireta sobre o tema.
    - Especifique qual parte do documento trata do assunto, mesmo que a resposta não seja conclusiva.
    - Use linguagem precisa e técnica, mas sempre com clareza.

    IMPORTANTE: Você deve entender o contexto jurídico brasileiro e a terminologia legal, especialmente em relação a:
    
    ANÁLISE DE PETIÇÃO INICIAL E CONTESTAÇÃO:
    - Identifique e separe claramente as RAZÕES DE FATO apresentadas
    - Identifique e separe claramente as RAZÕES DE DIREITO invocadas
    - Liste todos os PEDIDOS formulados de forma organizada
    - Identifique todos os DOCUMENTOS JUNTADOS mencionados
    - Destaque os PONTOS CONTROVERTIDOS de fato e direito
    
    ANÁLISE PROCESSUAL:
    - Verifique a validade das INTIMAÇÕES realizadas
    - Identifique DECISÕES já proferidas pelo juiz
    - Verifique PRAZOS PROCESSUAIS em curso
    - Identifique AUDIÊNCIAS designadas e suas finalidades
    - Verifique se foi determinada PERÍCIA e em que fase está
    
    TRABALHISTA:
    - Reclamante/Autor: trabalhador que move a ação
    - Reclamada/Réu: empregador contra quem a ação é movida
    - Verbas rescisórias: valores devidos na rescisão do contrato
    - Horas extras: trabalho além da jornada normal
    - Adicional de insalubridade/periculosidade: acréscimos por condições nocivas
    
    PREVIDENCIÁRIO:
    - Segurado: pessoa que contribui para a previdência
    - INSS: Instituto Nacional do Seguro Social (réu nas ações)
    - DIB: Data de Início do Benefício
    - DER: Data de Entrada do Requerimento
    - RMI: Renda Mensal Inicial do benefício
    - Carência: número mínimo de contribuições exigidas
    - Tempo de contribuição: períodos de recolhimento para a previdência
    
    Exemplos do formato esperado:
    Pergunta: "Quais são as razões de fato apresentadas na petição inicial?"
    Resposta: "RAZÕES DE FATO APRESENTADAS NA PETIÇÃO INICIAL:
    1. O autor foi admitido pela empresa ré em 10/01/2019 para exercer a função de auxiliar administrativo
    2. Em 15/03/2021, o autor foi demitido sem justa causa
    3. A empresa não pagou as verbas rescisórias devidas
    4. O autor trabalhava em jornada extraordinária sem receber horas extras
    5. O autor sofreu assédio moral por parte do supervisor direto"
    
    Pergunta: "Quais são os pedidos formulados na petição inicial?"
    Resposta: "PEDIDOS FORMULADOS NA PETIÇÃO INICIAL:
    1. Pagamento das verbas rescisórias no valor de R$ 5.432,10
    2. Pagamento de horas extras e reflexos no valor de R$ 12.345,67
    3. Indenização por danos morais no valor de R$ 20.000,00
    4. Aplicação da multa do art. 477 da CLT
    5. Condenação da ré ao pagamento de honorários advocatícios de 15%"

    Responda de forma objetiva, clara e precisa, considerando o contexto específico do documento analisado.

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
        # Verifica se há metadados significativos
        significant_metadata = {k: v for k, v in metadata.items() 
                               if k != "text_sample" and v and 
                               (not isinstance(v, str) or len(v) > 3) and
                               not any(error in v for error in ["ao pagamento", "como micro", "e procurador"])}
        
        if significant_metadata:
            # Cria uma mensagem em formato de parágrafos para os metadados
            metadata_msg = "## 📑 Informações Extraídas do Documento\n\n"
            
            # Prioriza informações mais importantes
            priority_fields = {
                "Partes do Processo": {
                    "autor": "Autor/Reclamante",
                    "reu": "Réu/Reclamada",
                    "cpf_reclamante": "CPF do Autor",
                    "cnpj_reclamada": "CNPJ do Réu"
                },
                "Informações Processuais": {
                    "tipo_acao": "Tipo de Ação",
                    "numero_processo": "Número do Processo",
                    "valor_causa": "Valor da Causa",
                    "vara": "Vara/Tribunal"
                },
                "Representantes": {
                    "advogado_autor": "Advogado do Autor",
                    "advogado_reu": "Advogado do Réu",
                    "oab_advogado": "OAB",
                    "juiz": "Juiz"
                },
                "Datas Importantes": {
                    "data_autuacao": "Data de Autuação",
                    "data_admissao": "Data de Admissão",
                    "data_demissao": "Data de Demissão",
                    "data_inicio_beneficio": "DIB",
                    "data_entrada_requerimento": "DER"
                },
                "Informações Trabalhistas": {
                    "salario": "Salário",
                    "funcao": "Função",
                    "jornada": "Jornada"
                },
                "Informações Previdenciárias": {
                    "numero_beneficio": "Número do Benefício",
                    "especie_beneficio": "Espécie",
                    "renda_mensal_inicial": "RMI",
                    "periodos_contribuicao": "Períodos de Contribuição"
                }
            }
            
            # Constrói a mensagem por categoria prioritária
            for category, fields in priority_fields.items():
                category_content = ""
                
                for key, display_name in fields.items():
                    if key in significant_metadata and significant_metadata[key]:
                        value = significant_metadata[key]
                        
                        if isinstance(value, list):
                            category_content += f"**{display_name}:** {', '.join(value)}\n\n"
                        else:
                            category_content += f"**{display_name}:** {value}\n\n"
                
                if category_content:
                    metadata_msg += f"### {category}\n\n{category_content}"
            
            # Adiciona outros campos que não estão nas categorias prioritárias
            other_fields = ""
            for key, value in significant_metadata.items():
                if not any(key in fields for fields in priority_fields.values()):
                    display_key = key.replace('_', ' ').title()
                    
                    if isinstance(value, list):
                        other_fields += f"**{display_key}:** {', '.join(value)}\n\n"
                    else:
                        other_fields += f"**{display_key}:** {value}\n\n"
            
            if other_fields:
                metadata_msg += f"### Outras Informações\n\n{other_fields}"
            
            # Envia a mensagem com os metadados
            if len(metadata_msg) > 50:  # Se tem conteúdo além do título
                await cl.Message(content=metadata_msg).send()
            else:
                await cl.Message(content="⚠️ Não foi possível extrair metadados estruturados do documento.").send()
        else:
            # Extrai informações básicas como fallback
            await extract_basic_information(metadata.get("text_sample", ""))
    else:
        await cl.Message(content="⚠️ Nenhum metadado foi detectado automaticamente.").send()
        
async def extract_basic_information(text_sample):
    """Extrai informações básicas do texto como fallback quando a extração estruturada falha"""
    fallback_msg = "### Informações Detectadas no Documento\n\n"
    
    # Extrai datas
    datas = re.findall(r'\d{2}/\d{2}/\d{4}', text_sample)
    if datas:
        unique_datas = list(set(datas))[:5]  # Limita a 5 datas únicas
        fallback_msg += "**Datas encontradas:**\n"
        for data in unique_datas:
            fallback_msg += f"- {data}\n"
        fallback_msg += "\n"
    
    # Extrai valores monetários
    valores_pattern = r'R\$\s*([\d.,]+)'
    valores = re.findall(valores_pattern, text_sample)
    if valores:
        unique_valores = list(set(valores))[:5]  # Limita a 5 valores únicos
        fallback_msg += "**Valores monetários:**\n"
        for valor in unique_valores:
            # Normaliza o formato do valor
            if ',' in valor and '.' in valor:
                # Formato brasileiro com separador de milhar (R$ 1.234,56)
                valor_normalizado = valor.replace('.', '').replace(',', '.')
            elif ',' in valor:
                # Formato brasileiro sem separador de milhar (R$ 1234,56)
                valor_normalizado = valor.replace(',', '.')
            else:
                valor_normalizado = valor
            
            fallback_msg += f"- R$ {valor_normalizado}\n"
        fallback_msg += "\n"
    
    # Extrai possíveis números de processo/benefício
    numeros_pattern = r'\b\d{7}[-.]?\d{2}[-.]?\d{4}[-.]?\d[-.]?\d{4}\b|\b\d{10,13}\b'
    numeros = re.findall(numeros_pattern, text_sample)
    if numeros:
        unique_numeros = list(set(numeros))[:3]  # Limita a 3 números únicos
        fallback_msg += "**Possíveis números de processo/benefício:**\n"
        for numero in unique_numeros:
            fallback_msg += f"- {numero}\n"
        fallback_msg += "\n"
    
    # Extrai possíveis nomes de pessoas jurídicas
    empresas_pattern = r'\b([A-Z][A-ZÀ-Ú\s]+(?:LTDA|ME|EPP|S[/.]A[.]?|EIRELI))\b'
    empresas = re.findall(empresas_pattern, text_sample)
    if empresas:
        unique_empresas = list(set(empresas))[:3]  # Limita a 3 empresas únicas
        fallback_msg += "**Possíveis pessoas jurídicas:**\n"
        for empresa in unique_empresas:
            fallback_msg += f"- {empresa}\n"
        fallback_msg += "\n"
    
    # Extrai possíveis CPF/CNPJ
    documentos_pattern = r'\b\d{2}[.-]?\d{3}[.-]?\d{3}[/]?\d{4}[-]?\d{2}\b|\b\d{3}[.-]?\d{3}[.-]?\d{3}[-]?\d{2}\b'
    documentos = re.findall(documentos_pattern, text_sample)
    if documentos:
        unique_documentos = list(set(documentos))[:3]  # Limita a 3 documentos únicos
        fallback_msg += "**Possíveis CPF/CNPJ:**\n"
        for documento in unique_documentos:
            fallback_msg += f"- {documento}\n"
        fallback_msg += "\n"
    
    if len(fallback_msg) > 35:  # Se encontrou algo além do título
        await cl.Message(content=fallback_msg).send()
    else:
        await cl.Message(content="⚠️ **Não foi possível extrair informações detalhadas deste documento.**\n\nO sistema tentou analisar o documento, mas não encontrou dados estruturados suficientes.").send()

# Função auxiliar para limpar dados da sessão do usuário de forma segura
async def reset_user_session():
    keys = ["chain", "retriever", "original_texts", "normalized_texts", "metadatas", "metadata"]
    for key in keys:
        if cl.user_session.get(key) is not None:
            cl.user_session.set(key, None)

@cl.on_chat_start
async def on_chat_start():

    elements = [cl.Image(name="image1", display="inline", path="./image1PeritoDoc.jpg")]
    await cl.Message(content="Olá! Bem-vindo ao Assistente Jurídico Trabalhista e Previdenciário! Envie um documento para começar. 🤖", elements=elements).send()

    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Por favor, envie um arquivo PDF para começarmos.",
            accept=["application/pdf"],
            max_size_mb=80,
            timeout=180,
        ).send()

    file = files[0]
    msg = cl.Message(content=f"Processando `{file.name}`...")
    await msg.send()

    try:
        with open(file.path, "rb") as f:
            pdf_bytes = f.read()
        
        # Usa o método de extração híbrida
        pdf_text = process_pdf_with_hybrid_extraction(pdf_bytes)
        source_method = "Híbrido"
        
        if not pdf_text.strip():
            raise ValueError("Não foi possível extrair texto do documento.")
    except Exception as e:
        logging.error(f"[PDF EXTRACTION ERROR] {e}")
        await cl.Message(content=f"Erro ao processar o PDF: {str(e)}").send()
        return

    document_type = detect_document_type(pdf_text)
    cl.user_session.set("document_type", document_type)

    # Inicializa a memória de conversa
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    global extracted_metadata
    extracted_metadata = extract_explicit_metadata(pdf_text)
    logging.info(f"[METADATA EXTRAIDA] {extracted_metadata}")

    # Análises específicas por tipo de documento
    document_analysis = {}
    if document_type in ["ctps", "holerite", "contrato_trabalho"]:
        document_analysis = analyze_labor_document(pdf_text, document_type)
        logging.info(f"[ANÁLISE TRABALHISTA] {document_analysis}")
    elif document_type in ["cnis", "ppp", "carta_concessao", "guia_recolhimento"]:
        document_analysis = analyze_social_security_document(pdf_text, document_type)
        logging.info(f"[ANÁLISE PREVIDENCIÁRIA] {document_analysis}")
    elif document_type in ["contestação", "recurso", "petição_inicial"]:
        document_analysis = extract_arguments_from_document(pdf_text, document_type)
        logging.info(f"[ANÁLISE DE ARGUMENTOS] {document_analysis}")
        
        # Extrai pontos controvertidos para documentos processuais
        controversial_points = extract_controversial_points(pdf_text, document_type)
        document_analysis["pontos_controvertidos"] = controversial_points
        logging.info(f"[PONTOS CONTROVERTIDOS] {controversial_points}")
        
        # Extrai decisões judiciais
        judicial_decisions = extract_judicial_decisions(pdf_text)
        if judicial_decisions:
            document_analysis["decisoes_judiciais"] = judicial_decisions
            logging.info(f"[DECISÕES JUDICIAIS] {judicial_decisions}")
        
        # Extrai prazos processuais
        procedural_deadlines = extract_procedural_deadlines(pdf_text)
        if procedural_deadlines:
            document_analysis["prazos_processuais"] = procedural_deadlines
            logging.info(f"[PRAZOS PROCESSUAIS] {procedural_deadlines}")
        
        # Extrai audiências
        hearings = extract_hearings(pdf_text)
        if hearings:
            document_analysis["audiencias"] = hearings
            logging.info(f"[AUDIÊNCIAS] {hearings}")
    
    # Adiciona análises específicas aos metadados
    if document_analysis:
        extracted_metadata.update({"analise_especializada": document_analysis})

    pdf_text = pdf_text.replace("-\n", "").replace("\n", " ")
    original_chunks = text_splitter.split_text(pdf_text)

    # Adiciona marcação extra para trechos com instruções futuras
    for i, chunk in enumerate(original_chunks):
        if any(kw in chunk.lower() for kw in ["deverá informar", "com antecedência de", "designar audiência", "será designada", "intimar"]):
            original_chunks[i] = "[INSTRUÇÃO FUTURA] " + chunk
        # Adiciona marcação para argumentos importantes
        if any(kw in chunk.lower() for kw in ["improcedente", "procedente", "prescrição", "incompetência", "ilegitimidade"]):
            original_chunks[i] = "[ARGUMENTO RELEVANTE] " + chunk
        # Adiciona marcação para valores e cálculos
        if any(kw in chunk.lower() for kw in ["valor de", "cálculo", "montante", "quantia", "rmi", "salário-de-contribuição"]):
            original_chunks[i] = "[VALOR RELEVANTE] " + chunk
        # Adiciona marcação para pontos controvertidos
        if any(kw in chunk.lower() for kw in ["controvertido", "controvérsia", "divergência", "discordância"]):
            original_chunks[i] = "[PONTO CONTROVERTIDO] " + chunk
        # Adiciona marcação para decisões judiciais
        if any(kw in chunk.lower() for kw in ["decido", "defiro", "indefiro", "determino", "homologo", "julgo"]):
            original_chunks[i] = "[DECISÃO JUDICIAL] " + chunk

    normalized_texts = [normalize_text(t) for t in original_chunks]
    metadatas = [{"source": f"Trecho {i+1}"} for i in range(len(normalized_texts))]

    logging.info(f"[EXTRACTION] Método: {source_method}, Chunks gerados: {len(original_chunks)}")

    # Antes de tudo, limpe a sessão
    await reset_user_session()

    # Depois de carregar e processar o PDF:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    docsearch = await cl.make_async(FAISS.from_texts)(
        normalized_texts, embeddings, metadatas=metadatas
    )
    
    # Modifique o retriever para usar parâmetros mais simples
    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 12}
    )

    # Cria a cadeia de conversação
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

    # Salva tudo na sessão atual
    cl.user_session.set("chain", chain)
    cl.user_session.set("retriever", retriever)
    cl.user_session.set("original_texts", original_chunks)
    cl.user_session.set("normalized_texts", normalized_texts)
    cl.user_session.set("metadatas", metadatas)
    cl.user_session.set("memory", memory)
    cl.user_session.set("document_analysis", document_analysis)

    # Prepara mensagem de feedback baseada no tipo de documento
    feedback_msg = ""
    if document_type == "petição_inicial":
        feedback_msg = "✅ Petição inicial identificada! Posso ajudar a extrair pedidos, fatos e fundamentos para preparar sua contestação."
    elif document_type == "contestação":
        feedback_msg = "✅ Contestação identificada! Posso ajudar a extrair argumentos da parte contrária para preparar sua réplica ou recurso."
    elif document_type == "recurso":
        feedback_msg = "✅ Recurso identificado! Posso ajudar a extrair argumentos recursais para preparar contrarrazões ou recurso adesivo."
    elif document_type == "sentença":
        feedback_msg = "✅ Sentença identificada! Posso ajudar a analisar os fundamentos da decisão para avaliar chances de recurso."
    elif document_type == "ctps":
        feedback_msg = "✅ Carteira de Trabalho identificada! Posso ajudar a extrair vínculos empregatícios, períodos e anotações."
    elif document_type == "holerite":
        feedback_msg = "✅ Holerite identificado! Posso ajudar a analisar remunerações, adicionais e descontos."
    elif document_type == "cnis":
        feedback_msg = "✅ CNIS identificado! Posso ajudar a analisar períodos contributivos, carência e vínculos."
    elif document_type == "ppp":
        feedback_msg = "✅ PPP identificado! Posso ajudar a analisar exposição a agentes nocivos para aposentadoria especial."
    elif document_type == "carta_concessao":
        feedback_msg = "✅ Carta de Concessão identificada! Posso ajudar a analisar dados do benefício concedido."
    else:
        feedback_msg = f"✅ Documento processado com sucesso via `{source_method}`! Posso ajudar a extrair informações relevantes."

    msg.content = f"Processamento de `{file.name}` concluído! {feedback_msg}"
    await msg.update()

    # Mensagem de orientação final com sugestões específicas para análise jurídica
    await cl.Message(content="🔍 **Como posso ajudar?** Você pode me perguntar sobre:\n\n"
                            "- **Análise de Petição Inicial e Contestação:**\n"
                            "  - Razões de Fato apresentadas\n"
                            "  - Razões de Direito invocadas\n"
                            "  - Pedidos formulados\n"
                            "  - Documentos juntados\n"
                            "  - Pontos controvertidos de fato e direito\n\n"
                            "- **Análise Processual:**\n"
                            "  - Validade das intimações\n"
                            "  - Decisões já proferidas pelo juiz\n"
                            "  - Prazos processuais\n"
                            "  - Andamento processual\n\n"
                            "- **Análise Documental:**\n"
                            "  - Extrair argumentos para contestação ou recurso\n"
                            "  - Analisar períodos de contribuição e carência\n"
                            "  - Verificar valores de remuneração ou benefícios\n"
                            "  - Identificar fundamentos legais e jurisprudência").send()

@cl.on_message
async def main(message: str):
    chain = cl.user_session.get("chain")
    if chain is None:
        await cl.Message(content="⚠️ Cadeia não inicializada. Envie um documento para começar.").send()
        return

    # Log detalhado da pergunta e contexto
    logging.info(f"[PERGUNTA ORIGINAL] {message.content}")
    document_type = cl.user_session.get("document_type", "desconhecido")
    logging.info(f"[TIPO DE DOCUMENTO] {document_type}")

    query = normalize_text(message.content)
    
    try:
        docs = await chain.retriever.ainvoke(query)

        # Log dos documentos recuperados
        logging.info(f"[DOCUMENTOS RECUPERADOS] Total: {len(docs)}")
        for i, doc in enumerate(docs[:3]):
            logging.info(f"[DOC {i+1}] Fonte: {doc.metadata.get('source', '?')}")
            logging.info(f"[DOC {i+1}] Conteúdo: {doc.page_content[:150]}...")

        # Verifica se a pergunta é sobre análise jurídica específica
        is_fact_analysis = any(term in message.content.lower() for term in 
                              ["razões de fato", "fatos", "aconteceu", "ocorreu"])
        
        is_law_analysis = any(term in message.content.lower() for term in 
                             ["razões de direito", "direito", "fundamentos", "legislação"])
        
        is_request_analysis = any(term in message.content.lower() for term in 
                                 ["pedidos", "requer", "tutela", "liminar"])
        
        is_document_analysis = any(term in message.content.lower() for term in 
                                  ["documentos", "provas", "juntados", "anexos"])
        
        is_controversy_analysis = any(term in message.content.lower() for term in 
                                     ["pontos controvertidos", "controvérsia", "divergência"])
        
        is_process_analysis = any(term in message.content.lower() for term in 
                                 ["intimações", "decisão", "audiência", "perícia", "prazo"])

        # Reordenação semântica dos documentos
        docs = rerank_semantically(message.content, docs)

        if not docs:
            query_terms = " ".join([word for word in message.content.lower().split() if len(word) > 3])
            fallback_docs = await chain.retriever.ainvoke(query_terms)
            
            if fallback_docs:
                docs = fallback_docs[:3]  # Use os 3 primeiros documentos da busca de fallback
                logging.info(f"[FALLBACK] Usando busca alternativa com termos: {query_terms}")
            else:
                # Resposta amigável quando não há documentos relevantes
                await cl.Message(content="Não encontrei informações específicas sobre isso no documento. O documento fornecido parece não conter detalhes sobre o que você está perguntando. Posso ajudar com outras informações que estejam presentes no documento?").send()
                return

        # Limita o número de documentos para evitar respostas muito grandes
        if len(docs) > 3:
            docs = docs[:3]
            
        logging.info("[TRECHOS USADOS] " + "; ".join([doc.metadata.get("source", "?") for doc in docs]))
        
        start_time = datetime.now()
        logging.info(f"[PERGUNTA RECEBIDA] {message.content}")
        
        # SOLUÇÃO: Abordagem direta para evitar problemas com chaves ausentes
        try:
            # Cria um contexto combinado a partir dos documentos
            context_text = "\n\n".join([doc.page_content for doc in docs])
            
            # Cria um prompt adaptativo sem depender de metadados problemáticos
            prompt = build_adaptive_prompt(message.content, {})
            
            # Cria uma instância do modelo de linguagem
            llm = ChatOpenAI(model="gpt-4", temperature=0)
            
            # Prepara os dados de entrada
            input_data = {
                "question": message.content,
                "context": context_text
            }
            
            # Invoca o modelo diretamente
            response = await llm.ainvoke(
                prompt.format_prompt(**input_data).to_messages()
            )
            
            # Extrai a resposta
            answer = response.content
            
            response_time = (datetime.now() - start_time).total_seconds()
            logging.info(f"[TEMPO DE RESPOSTA] {response_time:.2f} segundos")
            
            # Formata a resposta para perguntas específicas de análise jurídica
            if is_fact_analysis:
                if not answer.startswith("RAZÕES DE FATO") and ":" not in answer[:30]:
                    answer = "RAZÕES DE FATO APRESENTADAS:\n\n" + answer
            
            elif is_law_analysis:
                if not answer.startswith("RAZÕES DE DIREITO") and ":" not in answer[:30]:
                    answer = "RAZÕES DE DIREITO INVOCADAS:\n\n" + answer
            
            elif is_request_analysis:
                if not answer.startswith("PEDIDOS") and ":" not in answer[:30]:
                    answer = "PEDIDOS FORMULADOS:\n\n" + answer
            
            elif is_document_analysis:
                if not answer.startswith("DOCUMENTOS") and ":" not in answer[:30]:
                    answer = "DOCUMENTOS JUNTADOS:\n\n" + answer
            
            elif is_controversy_analysis:
                if not answer.startswith("PONTOS CONTROVERTIDOS") and ":" not in answer[:30]:
                    answer = "PONTOS CONTROVERTIDOS:\n\n" + answer
            
            elif is_process_analysis:
                if not answer.startswith("ANÁLISE PROCESSUAL") and ":" not in answer[:30]:
                    answer = "ANÁLISE PROCESSUAL:\n\n" + answer
            
            # Adiciona numeração se não existir
            if any([is_fact_analysis, is_law_analysis, is_request_analysis, is_document_analysis, is_controversy_analysis]):
                lines = answer.split('\n')
                formatted_lines = []
                point_count = 1
                for line in lines:
                    if line.strip() and not line.startswith('#') and not line.startswith('-') and not line.startswith('*') and not re.search(r'^\d+\.', line):
                        formatted_lines.append(f"{point_count}. {line}")
                        point_count += 1
                    else:
                        formatted_lines.append(line)
                answer = '\n'.join(formatted_lines)
            
            # Salva no histórico
            save_chat_history(message.content, answer)
            
            # Envia a resposta sem elementos adicionais
            await cl.Message(content=answer).send()
            
            # Sugestões de perguntas de acompanhamento baseadas no contexto
            # (código de sugestões permanece o mesmo)
            
        except Exception as e:
            # Captura erros específicos da invocação do modelo
            logging.error(f"[MODEL INVOCATION ERROR] {e}")
            
            # Tenta uma abordagem alternativa mais simples
            try:
                # Usa uma versão simplificada do prompt
                simple_prompt = f"""
                Você é um assistente jurídico. Analise o seguinte documento e responda à pergunta:
                
                Pergunta: {message.content}
                
                Documento:
                {context_text[:4000]}  # Limita o tamanho para evitar tokens excessivos
                """
                
                response = await llm.ainvoke([{"role": "user", "content": simple_prompt}])
                answer = response.content
                
                await cl.Message(content=answer).send()
                save_chat_history(message.content, answer)
                
            except Exception as e2:
                logging.error(f"[FALLBACK ERROR] {e2}")
                await cl.Message(content="Não consegui encontrar informações suficientes no documento para responder sua pergunta específica. O documento fornecido parece não conter os detalhes que você está buscando. Posso ajudar com outras informações que estejam presentes no documento?").send()
                
                # Sugestões específicas para análise jurídica
                suggestions = [
                    "Quais são as partes mencionadas no documento?",
                    "Qual é o tipo de procedimento ou ação em questão?",
                    "Há alguma data ou prazo importante mencionado?",
                    "Quais são os principais temas abordados no documento?"
                ]
                await cl.Message(content="**Você pode tentar perguntar sobre:**\n" + "\n".join([f"- {s}" for s in suggestions])).send()

    except Exception as e:
        # Captura erros gerais do processamento da mensagem
        logging.error(f"[GENERAL ERROR] {e}")
        
        # Resposta amigável para o usuário
        await cl.Message(content="Desculpe, tive dificuldade para processar sua pergunta. Pode tentar reformulá-la de outra maneira? Estou aqui para ajudar com informações contidas no documento.").send()
