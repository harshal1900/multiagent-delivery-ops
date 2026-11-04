"""Loads the delivery logs, customer/locker database and playbook, and
builds the Chroma vector store used for playbook retrieval.

Call load_all() once at process startup; it returns an IngestedData bundle
that tools.py and the LLM clients are built from.
"""
import sqlite3
from dataclasses import dataclass

import pandas as pd
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config


@dataclass
class IngestedData:
    delivery_logs_df: pd.DataFrame
    ground_truth_df: pd.DataFrame
    customers_df: pd.DataFrame
    lockers_df: pd.DataFrame
    playbook_chunks: list
    playbook_retriever: object
    db_conn: sqlite3.Connection


def load_delivery_logs_and_ground_truth() -> tuple[pd.DataFrame, pd.DataFrame]:
    delivery_logs_df = pd.read_csv(config.DELIVERY_LOGS_PATH, keep_default_na=False)
    ground_truth_df = pd.read_csv(config.GROUND_TRUTH_PATH, keep_default_na=False)
    return delivery_logs_df, ground_truth_df


def load_customers_and_lockers() -> tuple[pd.DataFrame, pd.DataFrame, sqlite3.Connection]:
    """Load the customers/lockers tables for inspection, and return a
    long-lived read-only connection the tools use directly."""
    load_conn = sqlite3.connect(config.CUSTOMERS_DB_PATH)
    customers_df = pd.read_sql_query("SELECT * FROM customers", load_conn)
    lockers_df = pd.read_sql_query("SELECT * FROM lockers", load_conn)
    load_conn.close()

    db_conn = sqlite3.connect(config.CUSTOMERS_DB_PATH)
    db_conn.row_factory = sqlite3.Row
    return customers_df, lockers_df, db_conn


def build_playbook_index() -> tuple[list, object]:
    """Load the playbook PDF, chunk it, and index it in an in-memory Chroma store."""
    pdf_loader = PyPDFLoader(config.PLAYBOOK_PDF_PATH)
    pdf_pages = pdf_loader.load()
    for doc in pdf_pages:
        doc.metadata["page"] = doc.metadata.get("page", 0) + 1

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    playbook_chunks = text_splitter.split_documents(pdf_pages)

    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    playbook_vectorstore = Chroma.from_documents(
        documents=playbook_chunks,
        embedding=embeddings,
        collection_name="playbook",
    )
    playbook_retriever = playbook_vectorstore.as_retriever(search_kwargs={"k": 3})
    return playbook_chunks, playbook_retriever


def load_all() -> IngestedData:
    delivery_logs_df, ground_truth_df = load_delivery_logs_and_ground_truth()
    customers_df, lockers_df, db_conn = load_customers_and_lockers()
    playbook_chunks, playbook_retriever = build_playbook_index()
    return IngestedData(
        delivery_logs_df=delivery_logs_df,
        ground_truth_df=ground_truth_df,
        customers_df=customers_df,
        lockers_df=lockers_df,
        playbook_chunks=playbook_chunks,
        playbook_retriever=playbook_retriever,
        db_conn=db_conn,
    )
