import time
import json
from sentence_transformers import SentenceTransformer, util
from rag_retriever import RAGRetriever
from rag_generator_gemini import RAGPipeline
from rouge_score import rouge_scorer
import sacrebleu

# Load Test Set

with open("data/test/test_questions.json", "r") as f:
    test_data = json.load(f)


# Load Components

retriever = RAGRetriever()
pipeline = RAGPipeline()
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)



#  1. RETRIEVAL METRICS
# Precision@K and Recall@K

def evaluate_retrieval(question, expected_answer, k=5):
    _, docs = retriever.build_context(question, k=k)
    if not docs:
        return 0.0, 0.0
    chunks = docs
    expected_emb = embed_model.encode(expected_answer, convert_to_tensor=True)
    retrieved_emb = embed_model.encode(chunks, convert_to_tensor=True)
    sims = util.cos_sim(expected_emb, retrieved_emb)[0]
    relevant_idx = int(sims.argmax())
    retrieved = list(range(k))
    relevant = [relevant_idx]
    precision = len(set(relevant) & set(retrieved)) / len(retrieved)
    recall = len(set(relevant) & set(retrieved)) / len(relevant)

    return precision, recall

#  2. GENERATION METRICS
# BLEU + ROUGE + (optional human evaluation)

def evaluate_generation(question, expected_answer):
    generated, _ = pipeline.generate_answer(question)
    bleu = sacrebleu.sentence_bleu(generated, [expected_answer]).score
    scores = rouge.score(expected_answer, generated)
    rouge1 = scores["rouge1"].fmeasure
    rougel = scores["rougeL"].fmeasure

    return generated, bleu, rouge1, rougel

#  3. LATENCY TEST
# Time taken per RAG query

def evaluate_latency(question):
    start = time.time()
    pipeline.generate_answer(question)
    end = time.time()
    return round(end - start, 3)

#  RUN ALL EVALUATIONS

print("       RAG Evaluation       ")
print("==============================\n")

for item in test_data:
    q = item["question"]
    ref = item["expected_answer"]

    print(f"QUESTION: {q}")
    print("-" * 50)

    # Retrieval
    precision, recall = evaluate_retrieval(q, ref)
    print(f"Retrieval Precision@5: {precision:.2f}")
    print(f"Retrieval Recall@5: {recall:.2f}")

    # Generation
    generated, bleu, r1, rl = evaluate_generation(q, ref)
    print(f"BLEU Score: {bleu:.2f}")
    print(f"ROUGE-1: {r1:.2f}")
    print(f"ROUGE-L: {rl:.2f}")

    # Latency
    latency = evaluate_latency(q)
    print(f"Latency: {latency} sec")

    print(f"Generated Answer:\n{generated}")
    print("\n" + "=" * 50 + "\n")

