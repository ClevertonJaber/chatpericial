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
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# Configuração de logging
logging.basicConfig(
    level=logging.INFO, 
    filename="chat_pericial.log", 
    filemode="a", 
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
)

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
    history_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input,
        "response": response
    }
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = []
    history.append(history_entry)
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

nlp = spacy.load("pt_core_news_md")
sbert_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

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

def detect_document_type(text: str) -> str:
    """Detecta o tipo de documento jurídico com base no conteúdo"""
    # Palavras-chave para cada tipo de documento
    keywords = {
        "petição_inicial": ["petição inicial", "autor requer", "dos pedidos", "dos fatos", "do direito", 
                           "deferimento", "termos em que", "pede deferimento"],
        "contestação": ["contestação", "preliminarmente", "mérito", "improcedente", "improcedência", 
                       "contesta", "contestar"],
        "recurso": ["recurso", "recorre", "reforma", "reformar", "decisão recorrida", "data venia",
                   "colenda", "egrégio", "recurso ordinário", "agravo de instrumento"],
        "sentença": ["sentença", "julgo", "dispositivo", "condeno", "improcedente", "procedente", 
                    "fundamentação", "relatório", "isto posto"],
        "ctps": ["carteira de trabalho", "ctps", "anotação", "registro de empregado", "admissão", "demissão"],
        "holerite": ["contracheque", "holerite", "folha de pagamento", "salário", "remuneração", 
                    "proventos", "descontos", "líquido a receber"],
        "contrato_trabalho": ["contrato de trabalho", "contrato individual", "prazo indeterminado", 
                             "regime de trabalho", "jornada", "remuneração"],
        "cnis": ["cnis", "cadastro nacional", "vínculos", "contribuições", "extrato previdenciário",
                "inss", "nis", "pis/pasep"],
        "ppp": ["perfil profissiográfico", "ppp", "agentes nocivos", "exposição", "insalubridade",
               "periculosidade", "aposentadoria especial"],
        "carta_concessao": ["carta de concessão", "benefício concedido", "rmi", "dib", "dip",
                           "der", "espécie", "nb"],
        "guia_recolhimento": ["gps", "guia da previdência", "contribuição", "recolhimento", 
                             "competência", "autônomo", "facultativo"]
    }
    
    # Conta ocorrências de palavras-chave
    counts = {doc_type: 0 for doc_type in keywords}
    text_lower = text.lower()
    
    for doc_type, terms in keywords.items():
        for term in terms:
            counts[doc_type] += text_lower.count(term)
    
    # Usa embeddings para classificação mais sofisticada
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
    
    # Calcula embeddings
    text_embedding = sbert_model.encode([text_lower[:1000]])[0]  # Usa apenas o início do texto
    desc_embeddings = sbert_model.encode(doc_descriptions)
    
    # Calcula similaridade
    similarities = cosine_similarity([text_embedding], desc_embeddings)[0]
    
    # Combina contagem de palavras-chave com similaridade semântica
    combined_scores = {
        doc_type: (counts[doc_type] * 0.7) + (similarities[i] * 0.3)
        for i, doc_type in enumerate(doc_types)
    }
    
    # Determina o tipo mais provável
    most_likely_type = max(combined_scores, key=combined_scores.get)
    confidence = combined_scores[most_likely_type]
    
    logging.info(f"[DOCUMENT TYPE] Detectado: {most_likely_type} (confiança: {confidence:.2f})")
    
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
    
    # Extrai períodos de contribuição do CNIS ou CTPS
    if document_type in ["cnis", "ctps"]:
        # Padrão para datas no formato DD/MM/AAAA a DD/MM/AAAA
        periodos = re.findall(r'(\d{2}/\d{2}/\d{4})\s*(?:a|até|-)\s*(\d{2}/\d{2}/\d{4})', text)
        for inicio, fim in periodos:
            # Tenta encontrar o empregador/empresa relacionado a este período
            contexto = text[max(0, text.find(inicio)-100):min(len(text), text.find(fim)+100)]
            empresa = ""
            for linha in contexto.split('\n'):
                if re.search(r'empresa|empregador|contratante', linha, re.IGNORECASE) and not inicio in linha and not fim in linha:
                    empresa = linha.strip()
                    break
            
            analysis["periodos_contribuicao"].append({
                "inicio": inicio,
                "fim": fim,
                "empresa": empresa
            })
        
        # Calcula tempo total se houver períodos
        if analysis["periodos_contribuicao"]:
            total_dias = 0
            for periodo in analysis["periodos_contribuicao"]:
                inicio = datetime.strptime(periodo["inicio"], "%d/%m/%Y")
                fim = datetime.strptime(periodo["fim"], "%d/%m/%Y")
                dias = (fim - inicio).days
                total_dias += dias
            
            anos = total_dias // 365
            meses = (total_dias % 365) // 30
            dias_restantes = (total_dias % 365) % 30
            
            analysis["tempo_total"] = f"{anos} anos, {meses} meses e {dias_restantes} dias"
            analysis["carencia"] = f"{len(analysis['periodos_contribuicao'])} competências"
    
    # Extrai informações sobre atividades especiais (PPP)
    if document_type == "ppp":
        # Busca por seções de agentes nocivos
        agentes_nocivos = re.findall(r'(?:Agente Nocivo|Agentes Nocivos|Fator de Risco)[:\s]+([\w\s,.;]+?)(?=\n)', text, re.IGNORECASE)
        for agente in agentes_nocivos:
            if len(agente.strip()) > 3:  # Ignora resultados muito curtos
                analysis["atividades_especiais"].append(agente.strip())
        
        # Busca por EPI
        epi_info = re.search(r'EPI[:\s]+([\w\s,.;]+?)(?=\n)', text, re.IGNORECASE)
        if epi_info:
            analysis["observacoes"].append(f"EPI: {epi_info.group(1).strip()}")
    
    # Extrai informações sobre benefícios (Carta de Concessão)
    if document_type == "carta_concessao":
        # Busca por número do benefício
        nb_match = re.search(r'(?:Benefício|NB)[:\s]+(\d{10})', text, re.IGNORECASE)
        if nb_match:
            analysis["beneficios_anteriores"].append({
                "nb": nb_match.group(1)
            })
            
            # Busca por DIB, DIP e RMI próximos ao NB encontrado
            contexto = text[max(0, text.find(nb_match.group(0))-200):min(len(text), text.find(nb_match.group(0))+500)]
            
            dib_match = re.search(r'DIB[:\s]+(\d{2}/\d{2}/\d{4})', contexto, re.IGNORECASE)
            if dib_match:
                analysis["beneficios_anteriores"][-1]["dib"] = dib_match.group(1)
                
            dip_match = re.search(r'DIP[:\s]+(\d{2}/\d{2}/\d{4})', contexto, re.IGNORECASE)
            if dip_match:
                analysis["beneficios_anteriores"][-1]["dip"] = dip_match.group(1)
                
            rmi_match = re.search(r'RMI[:\s]+R\$\s*([\d.,]+)', contexto, re.IGNORECASE)
            if rmi_match:
                analysis["beneficios_anteriores"][-1]["rmi"] = rmi_match.group(1)
                
            especie_match = re.search(r'Espécie[:\s]+(\d{2})', contexto, re.IGNORECASE)
            if especie_match:
                analysis["beneficios_anteriores"][-1]["especie"] = especie_match.group(1)
    
    # Extrai informações sobre recolhimentos (GPS)
    if document_type == "guia_recolhimento":
        # Busca por competências e valores
        recolhimentos = re.findall(r'(?:Competência|Comp)[:\s]+(\d{2}/\d{4}).*?(?:Valor|Total)[:\s]+R\$\s*([\d.,]+)', text, re.IGNORECASE | re.DOTALL)
        for competencia, valor in recolhimentos:
            analysis["observacoes"].append(f"Recolhimento {competencia}: R$ {valor}")
    
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
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def rerank_semantically(question: str, documents: list[Document]) -> list[Document]:
    """Reordena documentos com base na similaridade semântica com a pergunta"""
    # Expandir a pergunta com termos relacionados ao contexto jurídico trabalhista e previdenciário
    expanded_question = expand_question_for_legal_context(question)
    
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
    doc_embeddings = sbert_model.encode(doc_texts)
    question_embedding = sbert_model.encode([expanded_question])[0]
    
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
    metadata = {}

    # ----------- INFORMAÇÕES BÁSICAS -----------
    if match := re.search(r"Data da Autua[çc][aã]o[:\s]+(\d{2}/\d{2}/\d{4})", text, re.IGNORECASE):
        metadata["data_autuacao"] = match.group(1)
    if match := re.search(r"Valor da causa[:\s]+R\$\s*([\d.,]+)", text, re.IGNORECASE):
        metadata["valor_causa"] = match.group(1)
    if match := re.search(r"Processo\s*n[oº]?[:\s]*(\d{7}-\d{2}\.\d{4}\.\d{1,2}\.\d{4})", text):
        metadata["numero_processo"] = match.group(1)
    if match := re.search(r"Vara do Trabalho de ([\w\s]+)", text):
        metadata["vara"] = match.group(1).strip()

    # ----------- EXTRAÇÃO DE PARTES E ADVOGADOS -----------
    match_partes = re.search(
        r"(AUTOR|RECLAMANTE|SEGURADO)[:\s]+([^\n\r]+?)\s+ADVOGADO[:\s]+([^\n\r]+?)\s+(R[ÉE]U|RECLAMAD[OA]|INSS)[:\s]+([^\n\r]+?)\s+ADVOGADO[:\s]+([^\n\r]+)",
        text, re.IGNORECASE
    )
    if match_partes:
        metadata["autor"] = match_partes.group(1).strip()
        metadata["advogado_autor"] = match_partes.group(2).strip()
        metadata["reu"] = match_partes.group(3).strip()
        metadata["advogado_reu"] = match_partes.group(4).strip()
    else:
        linhas = text.splitlines()
        partes = {}
        for i, linha in enumerate(linhas):
            linha_norm = linha.strip().lower()
            if any(p in linha_norm for p in ["autor:", "reclamante:", "segurado:"]):
                partes["autor"] = linha.split(":", 1)[-1].strip()
            elif "advogado" in linha_norm and "autor" in partes and "advogado_autor" not in partes:
                partes["advogado_autor"] = linha.split(":", 1)[-1].strip()
            elif any(p in linha_norm for p in ["réu:", "reclamada:", "inss:"]):
                partes["reu"] = linha.split(":", 1)[-1].strip()
            elif "advogado" in linha_norm and "reu" in partes and "advogado_reu" not in partes:
                partes["advogado_reu"] = linha.split(":", 1)[-1].strip()
        metadata.update(partes)

        # ----------- EXTRAÇÃO DE DADOS TRABALHISTAS -----------
    if match := re.search(r"Data de Admissão[:\s]+(\d{2}/\d{2}/\d{4})", text, re.IGNORECASE):
        metadata["data_admissao"] = match.group(1)
    if match := re.search(r"Data de Demissão[:\s]+(\d{2}/\d{2}/\d{4})", text, re.IGNORECASE):
        metadata["data_demissao"] = match.group(1)
    if match := re.search(r"Salário[:\s]+R\$\s*([\d.,]+)", text, re.IGNORECASE):
        metadata["salario"] = match.group(1)
    if match := re.search(r"Função[:\s]+([\w\s]+)", text, re.IGNORECASE):
        metadata["funcao"] = match.group(1).strip()
    if match := re.search(r"Jornada[:\s]+([\w\s:àh]+)", text, re.IGNORECASE):
        metadata["jornada"] = match.group(1).strip()
    
    # ----------- EXTRAÇÃO DE DADOS PREVIDENCIÁRIOS -----------
    if match := re.search(r"NB[:\s]+(\d{10})", text, re.IGNORECASE):
        metadata["numero_beneficio"] = match.group(1)
    if match := re.search(r"DIB[:\s]+(\d{2}/\d{2}/\d{4})", text, re.IGNORECASE):
        metadata["data_inicio_beneficio"] = match.group(1)
    if match := re.search(r"DER[:\s]+(\d{2}/\d{2}/\d{4})", text, re.IGNORECASE):
        metadata["data_entrada_requerimento"] = match.group(1)
    if match := re.search(r"RMI[:\s]+R\$\s*([\d.,]+)", text, re.IGNORECASE):
        metadata["renda_mensal_inicial"] = match.group(1)
    if match := re.search(r"Espécie[:\s]+(\d{2})", text, re.IGNORECASE):
        metadata["especie_beneficio"] = match.group(1)
    
    # ----------- EXTRAÇÃO DE TEMPO DE CONTRIBUIÇÃO -----------
    tempo_contribuicao = re.findall(r"(\d{2}/\d{2}/\d{4})\s+a\s+(\d{2}/\d{2}/\d{4})", text)
    if tempo_contribuicao:
        metadata["periodos_contribuicao"] = [f"{inicio} a {fim}" for inicio, fim in tempo_contribuicao]

    # ----------- COMPLEMENTOS OPCIONAIS -----------
    if match := re.search(r"OAB[:/\s]*([A-Z]{2}\s*\d+)", text):
        metadata["oab_advogado"] = match.group(1)
    if match := re.search(r"CPF[:\s]*(\d{3}\.?\d{3}\.?\d{3}-?\d{2})", text, re.IGNORECASE):
        metadata["cpf_reclamante"] = match.group(1)
    if match := re.search(r"CNPJ[:\s]*(\d{2}\.?\d{3}\.?\d{3}/?0001-\d{2})", text, re.IGNORECASE):
        metadata["cnpj_reclamada"] = match.group(1)
    if match := re.search(r"PIS/PASEP[:\s]*(\d{3}\.?\d{5}\.?\d{2}-?\d{1})", text, re.IGNORECASE):
        metadata["pis_pasep"] = match.group(1)

    # ----------- EXTRAÇÃO COM spaCy (complementar) -----------
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PER" and any(term in ent.sent.text.lower() for term in ["juiz", "magistrado", "julgador"]):
            metadata["juiz"] = ent.text
        if ent.label_ == "LOC" and any(term in ent.sent.text.lower() for term in ["endereço", "localizado", "sede"]):
            metadata["endereco_relevante"] = ent.text
        if ent.label_ == "LAW" or any(term in ent.text.lower() for term in ["lei", "artigo", "decreto", "clt", "súmula"]):
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
    """Cria um prompt adaptativo baseado no tipo de pergunta do usuário"""
    metadata_str = format_metadata_for_prompt(metadata)
    
    # Detecta o tipo de pergunta
    query_lower = query.lower()
    
    # Instruções específicas baseadas no tipo de pergunta
    specific_instructions = ""
    
    if any(term in query_lower for term in ["autor", "reclamante", "requerente", "parte", "segurado"]):
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
    elif any(term in query_lower for term in ["valor", "causa", "condenação", "indenização", "dano", "salário", "remuneração", "rmi", "benefício"]):
        specific_instructions = """
        Ao responder sobre valores monetários:
        - Forneça apenas o valor e a que se refere
        - Destaque valores monetários em negrito usando **R$ 1.000,00**
        - Organize valores em lista se houver múltiplos valores
        - Seja extremamente conciso
        """
    elif any(term in query_lower for term in ["argumento", "contestação", "defesa", "recurso", "tese", "fundamento"]):
        specific_instructions = """
        Ao responder sobre argumentos jurídicos:
        - Enumere os argumentos identificados
        - Seja conciso e objetivo para cada argumento
        - Separe claramente preliminares de argumentos de mérito
        - Destaque fundamentos legais e jurisprudência citados
        - Organize a resposta em tópicos numerados
        """
    elif any(term in query_lower for term in ["tempo", "contribuição", "carência", "vínculo", "período", "aposentadoria"]):
        specific_instructions = """
        Ao responder sobre tempo de contribuição e carência:
        - Liste os períodos identificados com datas de início e fim
        - Calcule o tempo total se possível
        - Identifique períodos especiais se mencionados
        - Indique se a carência para benefícios foi cumprida
        - Organize a resposta em formato de lista
        """
    elif any(term in query_lower for term in ["jornada", "hora", "trabalho", "extra", "intervalo", "descanso"]):
        specific_instructions = """
        Ao responder sobre jornada de trabalho:
        - Indique horários de entrada e saída
        - Especifique dias da semana trabalhados
        - Mencione intervalos e períodos de descanso
        - Destaque informações sobre horas extras
        - Seja objetivo e preciso
        """

    system_template = f"""
    Você é um assistente jurídico especializado em direito trabalhista e previdenciário. Use as informações fornecidas no contexto para responder às perguntas do usuário sobre o documento enviado.

    Se a informação estiver presente no documento, forneça uma resposta direta e objetiva.
    
⚠️ Se a informação **não estiver explicitamente presente**, siga esta diretriz:
    - Explique se há previsão, instrução ou citação indireta sobre o tema.
    - Especifique qual parte do documento trata do assunto, mesmo que a resposta não seja conclusiva.
    - Use linguagem precisa e técnica, mas sempre com clareza.

    IMPORTANTE: Você deve entender o contexto jurídico brasileiro e a terminologia legal trabalhista e previdenciária:
    
    TRABALHISTA:
    - Identifique argumentos em contestações e recursos para elaborar defesas eficazes
    - Extraia informações relevantes de holerites, contratos de trabalho e CTPS
    - Analise jornadas de trabalho, horas extras, adicionais e verbas rescisórias
    - Identifique prescrição, prazos processuais e possíveis nulidades
    
    PREVIDENCIÁRIO:
    - Analise documentos como CTPS, CNIS, PPP, LTCAT, guias de recolhimento
    - Identifique períodos contributivos, carência e tempo de contribuição
    - Extraia informações sobre RMI, DIB, DIP, DER e espécies de benefícios
    - Verifique atividades especiais, insalubres ou perigosas
    
    Exemplos do formato esperado:
    Pergunta: "Quais os principais argumentos da contestação para elaborar as contrarrazões?"
    Resposta: "Argumentos principais da contestação:
    1. Prescrição quinquenal das verbas anteriores a 10/05/2018
    2. Inexistência de horas extras - alega compensação via banco de horas
    3. Adicional de insalubridade indevido - apresenta laudo técnico favorável
    4. Impugnação dos cálculos de verbas rescisórias - alega pagamento correto
    5. Jurisprudência citada: Súmula 85 do TST sobre compensação de jornada"
    
    Pergunta: "Qual o tempo de contribuição do segurado conforme CNIS?"
    Resposta: "Tempo de contribuição conforme CNIS:
    1. 05/03/1990 a 10/12/1995: 5 anos, 9 meses, 5 dias (empresa ABC Ltda)
    2. 02/02/1996 a 15/10/2010: 14 anos, 8 meses, 13 dias (empresa XYZ S/A)
    3. 01/12/2010 a 20/05/2022: 11 anos, 5 meses, 19 dias (contribuinte individual)
    Total: 31 anos, 11 meses, 7 dias de contribuição"

    Responda de forma objetiva, clara e precisa, considerando o contexto específico do direito trabalhista e previdenciário.

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
        # Cria uma mensagem em formato de parágrafos para os metadados
        metadata_msg = "## 📑 Informações Extraídas do Documento\n\n"
        
        # Organiza os metadados por categorias para melhor apresentação
        categories = {
            "Partes do Processo": ["autor", "reu", "reclamante", "reclamada", "segurado", "requerente"],
            "Representantes": ["advogado_autor", "advogado_reu", "oab_advogado", "procurador"],
            "Dados Pessoais": ["cpf_reclamante", "cnpj_reclamada", "rg", "data_nascimento"],
            "Informações Processuais": ["numero_processo", "vara", "juiz", "data_autuacao", "valor_causa", "tipo_acao"],
            "Localização": ["endereco_relevante", "comarca", "cidade", "estado"],
            "Legislação": ["leis_citadas", "fundamentos_legais", "jurisprudencia"],
            "Datas Importantes": ["data_admissao", "data_demissao", "data_audiencia", "prazo", "dib", "der", "dip"],
            "Valores": ["salario", "remuneracao", "rmi", "valor_beneficio", "verbas_rescisorias"],
            "Outros": []  # Categoria para itens que não se encaixam nas anteriores
        }
        
        # Organiza os metadados nas categorias
        categorized_metadata = {category: {} for category in categories}
        
        for key, value in metadata.items():
            if key == "analise_especializada":
                continue  # Trata separadamente
                
            # Determina a categoria para este metadado
            assigned = False
            for category, keywords in categories.items():
                if any(keyword in key for keyword in keywords):
                    categorized_metadata[category][key] = value
                    assigned = True
                    break
            
            # Se não foi atribuído a nenhuma categoria, coloca em "Outros"
            if not assigned:
                categorized_metadata["Outros"][key] = value
        
        # Constrói a mensagem por categoria
        for category, items in categorized_metadata.items():
            if items:  # Se há itens nesta categoria
                metadata_msg += f"### {category}\n\n"
                for key, value in items.items():
                    # Formata o nome da chave para exibição
                    display_key = key.replace('_', ' ').title()
                    
                    # Formata o valor para exibição completa
                    if isinstance(value, str) and len(value) > 0:
                        metadata_msg += f"**{display_key}:** {value}\n\n"
                    elif isinstance(value, list):
                        metadata_msg += f"**{display_key}:**\n"
                        for item in value:
                            metadata_msg += f"- {item}\n"
                        metadata_msg += "\n"
                    else:
                        metadata_msg += f"**{display_key}:** {value}\n\n"
        
        # Envia a mensagem com os metadados
        if metadata_msg != "## 📑 Informações Extraídas do Documento\n\n":
            await cl.Message(content=metadata_msg).send()
        else:
            await cl.Message(content="⚠️ Nenhum metadado básico foi detectado automaticamente.").send()
        
        # Trata a análise especializada separadamente
        if "analise_especializada" in metadata:
            analysis = metadata["analise_especializada"]
            
            # Verifica se a análise é um dicionário vazio ou com valores vazios
            if not analysis or all(not v for v in analysis.values()):
                await cl.Message(content="⚠️ **Não foi possível extrair informações especializadas deste documento.**\n\nO sistema tentou analisar o documento, mas não encontrou dados estruturados suficientes para uma análise detalhada.").send()
            else:
                # Formata a análise especializada em parágrafos
                analysis_msg = "## 📊 Análise Especializada do Documento\n\n"
                
                # Formata períodos de contribuição
                if "periodos_contribuicao" in analysis and analysis["periodos_contribuicao"]:
                    analysis_msg += "### Períodos de Contribuição\n\n"
                    for i, periodo in enumerate(analysis["periodos_contribuicao"], 1):
                        if isinstance(periodo, dict):
                            empresa = f" ({periodo.get('empresa')})" if periodo.get('empresa') else ""
                            analysis_msg += f"**Período {i}:** {periodo.get('inicio')} a {periodo.get('fim')}{empresa}\n\n"
                        else:
                            analysis_msg += f"**Período {i}:** {periodo}\n\n"
                
                # Formata tempo total
                if "tempo_total" in analysis and analysis["tempo_total"]:
                    analysis_msg += f"### Tempo Total\n\n**{analysis['tempo_total']}**\n\n"
                
                # Formata carência
                if "carencia" in analysis and analysis["carencia"]:
                    analysis_msg += f"### Carência\n\n**{analysis['carencia']}**\n\n"
                
                # Formata atividades especiais
                if "atividades_especiais" in analysis and analysis["atividades_especiais"]:
                    analysis_msg += "### Atividades Especiais\n\n"
                    for atividade in analysis["atividades_especiais"]:
                        analysis_msg += f"- {atividade}\n"
                    analysis_msg += "\n"
                
                # Formata benefícios anteriores
                if "beneficios_anteriores" in analysis and analysis["beneficios_anteriores"]:
                    analysis_msg += "### Benefícios Anteriores\n\n"
                    for i, beneficio in enumerate(analysis["beneficios_anteriores"], 1):
                        if isinstance(beneficio, dict):
                            analysis_msg += f"**Benefício {i}:** NB {beneficio.get('nb', 'N/A')}\n"
                            if beneficio.get('dib'):
                                analysis_msg += f"**DIB:** {beneficio['dib']}\n"
                            if beneficio.get('dip'):
                                analysis_msg += f"**DIP:** {beneficio['dip']}\n"
                            if beneficio.get('rmi'):
                                analysis_msg += f"**RMI:** R$ {beneficio['rmi']}\n"
                            if beneficio.get('especie'):
                                analysis_msg += f"**Espécie:** {beneficio['especie']}\n"
                            analysis_msg += "\n"
                        else:
                            analysis_msg += f"**Benefício {i}:** {beneficio}\n\n"
                
                # Formata RMI calculada
                if "rmi_calculada" in analysis and analysis["rmi_calculada"]:
                    analysis_msg += f"### RMI Calculada\n\n**R$ {analysis['rmi_calculada']}**\n\n"
                
                # Formata observações
                if "observacoes" in analysis and analysis["observacoes"]:
                    analysis_msg += "### Observações\n\n"
                    for obs in analysis["observacoes"]:
                        analysis_msg += f"- {obs}\n"
                    analysis_msg += "\n"
                
                # Formata dados de contrato de trabalho
                if "contrato" in analysis and analysis["contrato"]:
                    analysis_msg += "### Dados do Contrato\n\n"
                    for k, v in analysis["contrato"].items():
                        if v:  # Só mostra se tiver valor
                            analysis_msg += f"**{k.replace('_', ' ').title()}:** {v}\n\n"
                
                # Formata dados de remuneração
                if "remuneracao" in analysis and analysis["remuneracao"]:
                    analysis_msg += "### Dados de Remuneração\n\n"
                    for k, v in analysis["remuneracao"].items():
                        if v:  # Só mostra se tiver valor
                            analysis_msg += f"**{k.replace('_', ' ').title()}:** {v}\n\n"
                
                # Formata dados de jornada
                if "jornada" in analysis and analysis["jornada"]:
                    analysis_msg += "### Jornada de Trabalho\n\n"
                    for k, v in analysis["jornada"].items():
                        if v:  # Só mostra se tiver valor
                            analysis_msg += f"**{k.replace('_', ' ').title()}:** {v}\n\n"
                
                # Formata verbas rescisórias
                if "verbas_rescisoria" in analysis and analysis["verbas_rescisoria"]:
                    analysis_msg += "### Verbas Rescisórias\n\n"
                    for k, v in analysis["verbas_rescisoria"].items():
                        if v:  # Só mostra se tiver valor
                            analysis_msg += f"**{k.replace('_', ' ').title()}:** {v}\n\n"
                
                # Verifica se há algum conteúdo além do título
                if analysis_msg == "## 📊 Análise Especializada do Documento\n\n":
                    await cl.Message(content="⚠️ **Não foi possível extrair informações especializadas deste documento.**\n\nO sistema tentou analisar o documento, mas não encontrou dados estruturados suficientes para uma análise detalhada.").send()
                else:
                    await cl.Message(content=analysis_msg).send()
    else:
        await cl.Message(content="⚠️ Nenhum metadado foi detectado automaticamente.").send()

# Função auxiliar para limpar dados da sessão do usuário de forma segura
async def reset_user_session():
    keys = ["chain", "retriever", "original_texts", "normalized_texts", "metadatas", "metadata"]
    for key in keys:
        if cl.user_session.get(key) is not None:
            cl.user_session.set(key, None)

@cl.on_chat_start
async def on_chat_start():

    elements = [cl.Image(name="image1", display="inline", path="./robot.PNG")]
    await cl.Message(content="Olá! Bem-vindo ao Assistente Jurídico Trabalhista e Previdenciário! Envie um documento para começar. 🤖", elements=elements).send()

    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Por favor, envie um arquivo PDF para começarmos.",
            accept=["application/pdf"],
            max_size_mb=20,
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
    elif document_type in ["contestação", "recurso"]:
        document_analysis = extract_arguments_from_document(pdf_text, document_type)
        logging.info(f"[ANÁLISE DE ARGUMENTOS] {document_analysis}")
    
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

    # Mostra metadados extraídos
    await show_extracted_metadata(extracted_metadata)

    # Mostra análise especializada se disponível
    if document_analysis:
        analysis_msg = "## Análise Especializada do Documento\n\n"
        
        if document_type in ["ctps", "holerite", "contrato_trabalho"]:
            if document_analysis.get("contrato"):
                analysis_msg += "### Dados do Contrato\n"
                for k, v in document_analysis["contrato"].items():
                    analysis_msg += f"- **{k.replace('_', ' ').title()}**: {v}\n"
            
            if document_analysis.get("remuneracao"):
                analysis_msg += "\n### Dados de Remuneração\n"
                for k, v in document_analysis["remuneracao"].items():
                    analysis_msg += f"- **{k.replace('_', ' ').title()}**: {v}\n"
            
            if document_analysis.get("jornada"):
                analysis_msg += "\n### Jornada de Trabalho\n"
                for k, v in document_analysis["jornada"].items():
                    analysis_msg += f"- **{k.replace('_', ' ').title()}**: {v}\n"
            
            if document_analysis.get("verbas_rescisoria"):
                analysis_msg += "\n### Verbas Rescisórias\n"
                for k, v in document_analysis["verbas_rescisoria"].items():
                    analysis_msg += f"- **{k.replace('_', ' ').title()}**: {v}\n"
        
        elif document_type in ["cnis", "ppp", "carta_concessao", "guia_recolhimento"]:
            if document_analysis.get("periodos_contribuicao"):
                analysis_msg += "### Períodos de Contribuição\n"
                for i, periodo in enumerate(document_analysis["periodos_contribuicao"], 1):
                    if isinstance(periodo, dict):
                        empresa = f" ({periodo.get('empresa')})" if periodo.get('empresa') else ""
                        analysis_msg += f"- **Período {i}**: {periodo.get('inicio')} a {periodo.get('fim')}{empresa}\n"
                    else:
                        analysis_msg += f"- **Período {i}**: {periodo}\n"
            
            if document_analysis.get("tempo_total"):
                analysis_msg += f"\n### Tempo Total: **{document_analysis['tempo_total']}**\n"
            
            if document_analysis.get("carencia"):
                analysis_msg += f"\n### Carência: **{document_analysis['carencia']}**\n"
            
            if document_analysis.get("atividades_especiais"):
                analysis_msg += "\n### Atividades Especiais\n"
                for atividade in document_analysis["atividades_especiais"]:
                    analysis_msg += f"- {atividade}\n"
            
            if document_analysis.get("beneficios_anteriores"):
                analysis_msg += "\n### Benefícios Anteriores\n"
                for i, beneficio in enumerate(document_analysis["beneficios_anteriores"], 1):
                    analysis_msg += f"- **Benefício {i}**: NB {beneficio.get('nb', 'N/A')}\n"
                    if beneficio.get('dib'):
                        analysis_msg += f"  - DIB: {beneficio['dib']}\n"
                    if beneficio.get('dip'):
                        analysis_msg += f"  - DIP: {beneficio['dip']}\n"
                    if beneficio.get('rmi'):
                        analysis_msg += f"  - RMI: R$ {beneficio['rmi']}\n"
                    if beneficio.get('especie'):
                        analysis_msg += f"  - Espécie: {beneficio['especie']}\n"
        
        elif document_type in ["contestação", "recurso"]:
            if document_analysis.get("preliminares"):
                analysis_msg += "### Preliminares Identificadas\n"
                for i, preliminar in enumerate(document_analysis["preliminares"], 1):
                    analysis_msg += f"{i}. {preliminar}\n"
            
            if document_analysis.get("merito"):
                analysis_msg += "\n### Argumentos de Mérito\n"
                for i, argumento in enumerate(document_analysis["merito"], 1):
                    analysis_msg += f"{i}. {argumento}\n"
            
            if document_analysis.get("pedidos"):
                analysis_msg += "\n### Pedidos\n"
                for i, pedido in enumerate(document_analysis["pedidos"], 1):
                    analysis_msg += f"{i}. {pedido}\n"
            
            if document_analysis.get("fundamentos_legais"):
                analysis_msg += "\n### Fundamentos Legais\n"
                for fundamento in document_analysis["fundamentos_legais"]:
                    analysis_msg += f"- {fundamento}\n"
            
            if document_analysis.get("jurisprudencia"):
                analysis_msg += "\n### Jurisprudência Citada\n"
                for jurisprudencia in document_analysis["jurisprudencia"]:
                    analysis_msg += f"- {jurisprudencia}\n"
        
        await cl.Message(content=analysis_msg).send()

    # Mensagem de orientação final
    await cl.Message(content="🔍 **Como posso ajudar?** Você pode me perguntar sobre:\n\n"
                            "- Extrair argumentos para contestação ou recurso\n"
                            "- Analisar períodos de contribuição e carência\n"
                            "- Verificar valores de remuneração ou benefícios\n"
                            "- Identificar fundamentos legais e jurisprudência\n"
                            "- Calcular prazos e datas importantes\n"
                            "- Qualquer outra informação do documento").send()

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

        # Verifica se a pergunta é sobre extração de argumentos
        is_argument_extraction = any(term in message.content.lower() for term in 
                                ["argumento", "contestação", "defesa", "recurso", "contrarrazões", 
                                    "réplica", "tese", "fundamento"])
        
        # Verifica se a pergunta é sobre análise de tempo de contribuição
        is_contribution_analysis = any(term in message.content.lower() for term in 
                                    ["tempo", "contribuição", "carência", "vínculo", "cnis", 
                                    "aposentadoria", "benefício", "período"])
        
        # Verifica se a pergunta é sobre valores monetários
        is_value_analysis = any(term in message.content.lower() for term in 
                            ["valor", "salário", "remuneração", "rmi", "benefício", 
                            "verbas", "rescisórias", "indenização"])

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

        # Atualiza o prompt com base na pergunta atual e tipo de análise
        adaptive_prompt = build_adaptive_prompt(message.content, extracted_metadata)
        chain.combine_docs_chain.llm_chain.prompt = adaptive_prompt

        # Limita o número de documentos para evitar respostas muito grandes
        if len(docs) > 3:
            docs = docs[:3]
            
        logging.info("[TRECHOS USADOS] " + "; ".join([doc.metadata.get("source", "?") for doc in docs]))
        
        start_time = datetime.now()
        logging.info(f"[PERGUNTA RECEBIDA] {message.content}")
        
        # Verifica se o tipo de documento é compatível com a pergunta
        document_analysis = cl.user_session.get("document_analysis", {})
        
        # Mensagens personalizadas para incompatibilidade entre pergunta e tipo de documento
        if is_argument_extraction and document_type not in ["contestação", "recurso", "petição_inicial"]:
            await cl.Message(content="Este documento não parece conter argumentos jurídicos para contestação ou recurso. O documento foi identificado como " + 
                            f"**{document_type.replace('_', ' ').title()}**. " +
                            "Posso ajudar a extrair outras informações que estejam presentes neste tipo de documento.").send()
            return
            
        elif is_contribution_analysis and document_type not in ["cnis", "ctps", "ppp", "carta_concessao"]:
            await cl.Message(content="Este documento não parece conter informações sobre períodos de contribuição ou carência. O documento foi identificado como " + 
                            f"**{document_type.replace('_', ' ').title()}**. " +
                            "Posso ajudar a extrair outras informações que estejam presentes neste tipo de documento.").send()
            return
            
        # Adiciona contexto específico para perguntas especializadas (sem acessar chaves específicas)
        if is_argument_extraction and document_type in ["contestação", "recurso"]:
            context_msg = "Analisando argumentos da parte contrária para sua defesa..."
            await cl.Message(content=context_msg).send()
        
        elif is_contribution_analysis and document_type in ["cnis", "ctps", "ppp"]:
            context_msg = "Analisando períodos de contribuição e carência..."
            await cl.Message(content=context_msg).send()
        
        elif is_value_analysis and document_type in ["holerite", "carta_concessao"]:
            context_msg = "Analisando valores monetários do documento..."
            await cl.Message(content=context_msg).send()
        
        # Tenta invocar a cadeia com tratamento de erro
        try:
            res = await chain.ainvoke({"question": message.content})
            response_time = (datetime.now() - start_time).total_seconds()
            logging.info(f"[TEMPO DE RESPOSTA] {response_time:.2f} segundos")

            if isinstance(res, dict):
                answer = res.get("answer") or "Não encontrei informações específicas sobre isso no documento."
                
                # Formata a resposta para perguntas específicas
                if is_argument_extraction:
                    # Tenta estruturar melhor a resposta sobre argumentos
                    if not answer.startswith("Argumentos") and ":" not in answer[:30]:
                        answer = "Argumentos identificados:\n\n" + answer
                    
                    # Adiciona numeração se não existir
                    if not re.search(r'^\d+\.', answer.split('\n')[0]):
                        lines = answer.split('\n')
                        formatted_lines = []
                        point_count = 1
                        for line in lines:
                            if line.strip() and not line.startswith('#') and not line.startswith('-') and not line.startswith('*'):
                                if not re.search(r'^\d+\.', line):
                                    formatted_lines.append(f"{point_count}. {line}")
                                    point_count += 1
                                else:
                                    formatted_lines.append(line)
                            else:
                                formatted_lines.append(line)
                        answer = '\n'.join(formatted_lines)
                
                elif is_contribution_analysis:
                    # Tenta estruturar melhor a resposta sobre períodos contributivos
                    if "período" in answer.lower() and ":" not in answer[:30]:
                        answer = "Análise de períodos contributivos:\n\n" + answer
                
                elif is_value_analysis:
                    # Tenta estruturar melhor a resposta sobre valores
                    if "valor" in answer.lower() and ":" not in answer[:30]:
                        answer = "Análise de valores:\n\n" + answer
                    
                    # Destaca valores monetários
                    answer = re.sub(r'(R\$\s*[\d.,]+)', r'**\1**', answer)
                
                # Limita o tamanho da resposta
                if len(answer) > 1000:
                    answer = answer[:997] + "..."
                    
                # Salva no histórico
                save_chat_history(message.content, answer)
                
                # Envia a resposta sem elementos adicionais
                await cl.Message(content=answer).send()
                
                # Sugestões de perguntas de acompanhamento baseadas no contexto
                if is_argument_extraction:
                    suggestions = [
                        "Quais são os principais argumentos de defesa?",
                        "Quais fundamentos legais foram citados?",
                        "Existe alguma preliminar levantada?",
                        "Quais precedentes ou jurisprudência foram citados?"
                    ]
                    await cl.Message(content="**Perguntas sugeridas:**\n" + "\n".join([f"- {s}" for s in suggestions])).send()
                
                elif is_contribution_analysis:
                    suggestions = [
                        "Qual o tempo total de contribuição?",
                        "Quais períodos podem ser considerados especiais?",
                        "A carência para aposentadoria foi cumprida?",
                        "Existem períodos com inconsistências?"
                    ]
                    await cl.Message(content="**Perguntas sugeridas:**\n" + "\n".join([f"- {s}" for s in suggestions])).send()
                
                elif is_value_analysis:
                    suggestions = [
                        "Qual o valor total das verbas rescisórias?",
                        "Houve pagamento de horas extras?",
                        "Qual o valor do salário base?",
                        "Qual a RMI do benefício?"
                    ]
                    await cl.Message(content="**Perguntas sugeridas:**\n" + "\n".join([f"- {s}" for s in suggestions])).send()
                
            else:
                answer = str(res)
                if len(answer) > 1000:
                    answer = answer[:997] + "..."
                await cl.Message(content=answer.strip()).send()

            save_chat_history(message.content, answer)
            logging.info(f"[PERGUNTA] {message.content}")
            logging.info(f"[RESPOSTA] {answer[:200]}...")
            
        except Exception as e:
            # Captura erros específicos da invocação da cadeia
            logging.error(f"[CHAIN INVOCATION ERROR] {e}")
            
            # Resposta amigável para o usuário em vez de mostrar o erro técnico
            await cl.Message(content="Não consegui encontrar informações suficientes no documento para responder sua pergunta específica. O documento fornecido parece não conter os detalhes que você está buscando. Posso ajudar com outras informações que estejam presentes no documento?").send()
            
            # Sugestões genéricas
            suggestions = [
                "Qual o tipo de documento é este?",
                "Quais são as partes mencionadas no documento?",
                "Quais datas importantes são mencionadas?",
                "Quais valores são citados no documento?"
            ]
            await cl.Message(content="**Você pode tentar perguntar sobre:**\n" + "\n".join([f"- {s}" for s in suggestions])).send()

    except Exception as e:
        # Captura erros gerais do processamento da mensagem
        logging.error(f"[GENERAL ERROR] {e}")
        
        # Resposta amigável para o usuário
        await cl.Message(content="Desculpe, tive dificuldade para processar sua pergunta. Pode tentar reformulá-la de outra maneira? Estou aqui para ajudar com informações contidas no documento.").send()
            
            

