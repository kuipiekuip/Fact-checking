import json
from rank_bm25 import BM25Okapi

# Load the unified evidence corpus
with open('data\corpus_evidence_unified.json', 'r') as f:
    evidence_corpus = json.load(f)

# Load the claims
with open(r'data\train_claims_quantemp.json', 'r') as f:
    claims = json.load(f)

# Prepare the corpus for BM25
corpus = [evidence for evidence in evidence_corpus.values()]
tokenized_corpus = [doc.split() for doc in corpus]

# Initialize BM25
bm25 = BM25Okapi(tokenized_corpus)

# Function to retrieve top-k evidence
def retrieve_evidence(claim_text, k=5):
    tokenized_claim = claim_text.split()
    scores = bm25.get_scores(tokenized_claim)
    top_k_indices = scores.argsort()[-k:][::-1]
    top_k_documents = [corpus[idx] for idx in top_k_indices]
    return top_k_documents

# Limit to the first 20 claims for testing
claims_subset = claims[:20]

# Prepare data for NLI model
def prepare_nli_input(claim, evidence):
    nli_input = {
        'claim': claim,
        'evidence': " ".join(evidence)  # Concatenate top-k evidence into a single string
    }
    return nli_input

nli_data = []
for claim in claims_subset:
    claim_text = claim['claim']
    evidence = retrieve_evidence(claim_text)
    nli_input = prepare_nli_input(claim_text, evidence)
    nli_data.append(nli_input)

# Save prepared NLI data
with open('data/nli_input_20.json', 'w') as f:
    json.dump(nli_data, f)

# Print out the first few entries to check
for i, entry in enumerate(nli_data[:5]):
    print(f"Claim {i+1}: {entry['claim']}")
    print(f"Evidence {i+1}: {entry['evidence']}")
