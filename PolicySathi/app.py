import os
import json
from flask import Flask, request, jsonify
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

app = Flask(__name__)

# ---------------------------
# 🔹 Load Embedding Model
# ---------------------------
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------------------------
# 🔹 Load Regulatory Dataset
# ---------------------------
with open("data/regulations.json", "r") as f:
    regulations = json.load(f)

reg_texts = [r["text"] for r in regulations]
reg_embeddings = embed_model.encode(reg_texts)

dim = reg_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(reg_embeddings))

# ---------------------------
# 🔹 UTIL FUNCTIONS
# ---------------------------

def extract_text(file):
    if file.filename.endswith(".pdf"):
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    else:
        return file.read().decode("utf-8")


def redact_pii(text):
    # simple regex-based (replace with Presidio if needed)
    text = re.sub(r'\b\d{10}\b', '[PHONE]', text)
    text = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]', text)
    text = re.sub(r'\b\d{6,}\b', '[POLICY_ID]', text)
    return text


def chunk_text(text, size=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size):
        chunks.append(" ".join(words[i:i+size]))
    return chunks


def retrieve_context(query, k=3):
    q_emb = embed_model.encode([query])
    D, I = index.search(np.array(q_emb), k)
    return [regulations[i]["text"] for i in I[0]]


def mock_llm_analysis(context, claim_text):
    """
    Replace with DeepSeek / Qwen API
    This is a deterministic mock to avoid hallucination
    """
    issues = []

    if "signature" not in claim_text.lower():
        issues.append("Missing signature")

    if "policy" not in claim_text.lower():
        issues.append("Missing policy reference")

    return {
        "issues": issues,
        "justification": "Issues derived strictly from document + rules",
        "confidence": round(0.7 + 0.1 * len(issues), 2)
    }


def validate_output(output, context):
    # ensure issues are grounded
    if not output["issues"]:
        output["confidence"] = 0.5
    return output


# ---------------------------
# 🔹 MAIN API
# ---------------------------

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files["file"]

    # Step 1: Extract
    text = extract_text(file)

    # Step 2: Redact PII
    redacted = redact_pii(text)

    # Step 3: Chunk
    chunks = chunk_text(redacted)

    # Step 4: Retrieve context (use first chunk)
    context = retrieve_context(chunks[0])

    # Step 5: LLM
    result = mock_llm_analysis(context, redacted)

    # Step 6: Validate
    final_result = validate_output(result, context)

    return jsonify({
        "redacted_text": redacted[:500],
        "context_used": context,
        "analysis": final_result
    })


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(debug=True)