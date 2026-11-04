"""Runtime configuration loaded from environment variables.

Secrets never live in this repository. Copy .env.example to .env, fill in
real values, and this module (via python-dotenv) picks them up at import
time. The sample dataset under data/sample/ needs no credentials to load;
only the LLM calls do.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# Folder that holds customers.db, delivery_logs.csv, ground_truth.csv and
# exception_resolution_playbook.pdf. Defaults to the sample dataset shipped
# in this repo so the project runs out of the box in demo mode.
DATA_DIR = os.environ.get(
    "DATA_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample"),
)

DELIVERY_LOGS_PATH = os.path.join(DATA_DIR, "delivery_logs.csv")
GROUND_TRUTH_PATH = os.path.join(DATA_DIR, "ground_truth.csv")
CUSTOMERS_DB_PATH = os.path.join(DATA_DIR, "customers.db")
PLAYBOOK_PDF_PATH = os.path.join(DATA_DIR, "exception_resolution_playbook.pdf")

# OpenAI-compatible endpoint. OPENAI_API_BASE lets this point at a proxy
# (for example an academy or internal gateway) instead of api.openai.com.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "")
if OPENAI_API_BASE:
    os.environ["OPENAI_BASE_URL"] = OPENAI_API_BASE

GEN_MODEL = os.environ.get("GEN_MODEL", "gpt-4o")
EVAL_MODEL = os.environ.get("EVAL_MODEL", "gpt-4o")

# LangSmith tracing (optional). Leave LANGCHAIN_TRACING_V2 unset or "false"
# to run without a trace project.
LANGCHAIN_TRACING_V2 = os.environ.get("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.environ.get("LANGCHAIN_PROJECT", "multiagent-delivery-ops")

os.environ.setdefault("LANGCHAIN_TRACING_V2", LANGCHAIN_TRACING_V2)
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ.setdefault("LANGCHAIN_PROJECT", LANGCHAIN_PROJECT)

# Embedding model for the playbook vector store (downloaded once, cached
# locally by sentence-transformers).
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Resolution Agent / Communication Agent retry budgets and revision loop cap.
RESOLUTION_MAX_RETRIES = int(os.environ.get("RESOLUTION_MAX_RETRIES", "3"))
COMMUNICATION_MAX_RETRIES = int(os.environ.get("COMMUNICATION_MAX_RETRIES", "2"))
MAX_REVISION_LOOPS = int(os.environ.get("MAX_REVISION_LOOPS", "2"))
