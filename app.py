import re
import requests
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer("all-MiniLM-L6-v2")
classifier = joblib.load("hallucination_model.pkl")


def extract_entities(text):
    entities = set()

    for word in text.split():
        word = word.strip(".,!?;:()")

        if word and word[0].isupper():
            entities.add(word)

    return entities


def extract_claims(text):
    claims = []

    for sentence in re.split(r"[.!?]+", text):
        sentence = sentence.strip()

        if len(sentence.split()) >= 4:
            claims.append(sentence)

    return claims


def get_evidence(question):
    url = "https://en.wikipedia.org/w/api.php"

    headers = {
        "User-Agent": "AI-Hallucination-Risk-Predictor/1.0"
    }

    params = {
        "action": "query",
        "list": "search",
        "srsearch": question,
        "format": "json",
        "utf8": 1,
        "srlimit": 1
    }

    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=15
        )

        if response.status_code != 200:
            return ""

        data = response.json()
        results = data.get("query", {}).get("search", [])

        if not results:
            return ""

        title = results[0]["title"]

        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "exchars": 5000,
            "titles": title,
            "format": "json",
            "utf8": 1
        }

        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=15
        )

        if response.status_code != 200:
            return ""

        data = response.json()
        pages = data.get("query", {}).get("pages", {})

        for page in pages.values():
            evidence = page.get("extract", "")

            if evidence:
                return evidence

    except Exception:
        return ""

    return ""


def similarity(text1, text2):
    emb1 = model.encode([text1])
    emb2 = model.encode([text2])

    return cosine_similarity(emb1, emb2)[0][0]


def check_claim(claim, evidence, verified_answer):
    evidence_score = similarity(claim, evidence) * 100
    verified_score = similarity(claim, verified_answer) * 100

    claim_entities = extract_entities(claim)
    evidence_entities = extract_entities(evidence)

    entity_match = bool(
        claim_entities & evidence_entities
    )

    if evidence_score >= 75 and entity_match:
        result = "SUPPORTED"

    elif evidence_score >= 50 and entity_match:
        result = "POSSIBLY SUPPORTED"

    elif verified_score >= 70:
        result = "CONFLICTS WITH VERIFIED ANSWER"

    else:
        result = "UNSUPPORTED"

    return evidence_score, verified_score, entity_match, result


question = input("Enter your question: ")
ai_answer = input("Enter the AI-generated answer: ")
verified_answer = input("Enter the verified answer: ")


ai_entities = extract_entities(ai_answer)
verified_entities = extract_entities(verified_answer)
ai_claims = extract_claims(ai_answer)

print("\nRetrieving evidence...")

evidence = get_evidence(question)


semantic_similarity = similarity(
    ai_answer,
    verified_answer
) * 100


entity_mismatch = False

if ai_entities and verified_entities:
    entity_mismatch = ai_entities.isdisjoint(
        verified_entities
    )


claim_results = []

if evidence:

    for claim in ai_claims:

        score, verified_score, entity_match, result = check_claim(
            claim,
            evidence,
            verified_answer
        )

        claim_results.append({
            "claim": claim,
            "score": score,
            "verified_score": verified_score,
            "entity_match": entity_match,
            "result": result
        })


if claim_results:

    average_claim_score = sum(
        item["score"] for item in claim_results
    ) / len(claim_results)

else:
    average_claim_score = 0


if evidence:

    answer_evidence_similarity = similarity(
        ai_answer,
        evidence
    )

    prediction = classifier.predict(
        [[answer_evidence_similarity]]
    )[0]

    probabilities = classifier.predict_proba(
        [[answer_evidence_similarity]]
    )[0]

    ml_score = probabilities[1] * 100

else:
    prediction = None
    ml_score = 0


if evidence:

    similarity_risk = 100 - semantic_similarity
    claim_risk = 100 - average_claim_score

    risk_score = (
        ml_score * 0.50
        + similarity_risk * 0.25
        + claim_risk * 0.25
    )

else:

    risk_score = 100 - semantic_similarity


if entity_mismatch:
    risk_score += 10


unsupported_claims = sum(
    item["result"] == "UNSUPPORTED"
    for item in claim_results
)

conflicting_claims = sum(
    item["result"] == "CONFLICTS WITH VERIFIED ANSWER"
    for item in claim_results
)


risk_score += unsupported_claims * 5
risk_score += conflicting_claims * 10


risk_score = min(round(risk_score, 2), 100)


if risk_score <= 30:
    risk_level = "LOW"

elif risk_score <= 70:
    risk_level = "MEDIUM"

else:
    risk_level = "HIGH"


if not evidence:

    reason = "External evidence was not available."

elif conflicting_claims > 0:

    reason = "The AI answer conflicts with the verified answer."

elif unsupported_claims > 0:

    reason = "One or more claims are not supported by the available evidence."

elif entity_mismatch:

    reason = "Important entities in the AI answer differ from the verified answer."

elif prediction == 1:

    reason = "The model detected a higher hallucination risk."

else:

    reason = "The answer is largely consistent with the available evidence."


print("\n--- AI HALLUCINATION ANALYSIS ---")

print("Question:", question)
print("AI Answer:", ai_answer)
print("Verified Answer:", verified_answer)

print("\nAI Entities:", ai_entities)
print("Verified Entities:", verified_entities)

print("\nAI Claims:", ai_claims)

print("\nSemantic Similarity:",
      round(semantic_similarity, 2), "%")

print("Entity Mismatch:", entity_mismatch)


print("\n--- CLAIM VERIFICATION ---")

if claim_results:

    for item in claim_results:

        print("\nClaim:", item["claim"])

        print(
            "Evidence Similarity:",
            round(item["score"], 2),
            "%"
        )

        print(
            "Verified Answer Similarity:",
            round(item["verified_score"], 2),
            "%"
        )

        print("Entity Match:", item["entity_match"])
        print("Result:", item["result"])

else:

    print("No claims could be verified.")


if evidence:

    print(
        "\nAverage Claim-Evidence Similarity:",
        round(average_claim_score, 2),
        "%"
    )

    print(
        "ML Hallucination Probability:",
        round(ml_score, 2),
        "%"
    )

else:

    print(
        "\nML Hallucination Probability: Not available"
    )


print(
    "Final Hallucination Risk:",
    risk_score,
    "%"
)

print("Risk Level:", risk_level)
print("Reason:", reason)


if evidence:

    print("\n--- RETRIEVED EVIDENCE ---")
    print(evidence)

else:

    print("\nNo external evidence was retrieved.")