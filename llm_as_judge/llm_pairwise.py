import pandas as pd
import numpy as np
import os,re
from itertools import combinations
import torch
from collections import Counter
from transformers import pipeline,AutoTokenizer,AutoModelForCausalLM

INPUT_CSV="all_results.csv"
OUTPUT_CSV="results_scored.csv"
MODEL_PATH="./llama2/Llama-2-7b-chat-hf"

tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH)

model=AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.generation_config.do_sample=False
model.generation_config.temperature=None
model.generation_config.top_p=None

judge=pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1,
    return_full_text=False
)

MAX_CHARS = 720


def get_results(results):
    """
    if "Scored" not in results.columns:
        results["Scored"] = False

    results["Scored"] = results["Scored"].fillna(False)
    """

    available = results.loc[
        results["Scored"] == False,
        "Query"
    ].unique()

    if len(available) == 0:
        return None

    query = np.random.choice(available)

    ranks = (
        results.loc[
            results["Query"] == query,
            "Rank"
        ]
        .dropna()
        .unique()
    )

    rank = np.random.choice(ranks)
    
    subset = results[
    (results["Query"] == query) &
    (results["Rank"] == rank)
    ].copy()

    print(subset[["Method", "Rank"]])

    data = (
        subset
        .groupby("Method")
        .head(1)
        .copy()
    )

    print(f"Selected query={query}")
    print(f"Selected rank={rank}")
    print(f"Rows returned={len(data)}")

    return data

LABEL_PAIRS = [
    ("A", "B"),
    ("X", "Y")
]
def judge_pair(
    query,
    doc_a,
    doc_b,
    method_a,
    method_b,
    rank,
    comparisons,
    retries=5
):

    label1, label2 = LABEL_PAIRS[np.random.randint(len(LABEL_PAIRS))]

    # Randomize which document gets which label
    if np.random.rand() < 0.5:
        first_label, second_label = label1, label2
    else:
        first_label, second_label = label2, label1

    prompt = f"""
<s>[INST]

You are an expert information retrieval evaluator.

Your task is to determine which passage is MORE relevant to the query.

Important:
- Judge only semantic relevance.
- Ignore passage position.
- Labels are assigned randomly.
- Either passage may be better.

Query:
{query}

Passage {first_label}:
{doc_a[:MAX_CHARS]}

Passage {second_label}:
{doc_b[:MAX_CHARS]}

Respond with exactly one label. Do not give an explanation.

{label1}
or
{label2}

[/INST]
"""

    for attempt in range(retries):

        response = judge(
            prompt,
            max_new_tokens=1,
            do_sample=False
        )

        output = response[0]["generated_text"].strip()

        print("RAW:", repr(output))

        match = re.search(
            rf"\b({label1}|{label2})\b",
            output.upper()
        )

        if match:

            winning_label = match.group(1).upper()

            # Determine which method won
            if winning_label == first_label:
                winner_method = method_a
            else:
                winner_method = method_b

            # Record comparison
                
            comparisons.loc[len(comparisons)] = [
                query,
                rank,
                method_a,
                first_label,
                method_b,
                second_label,
                winner_method
            ]

        pd.DataFrame([{
            "Query": query,
            "Rank": rank,
            "Method_A": method_a,
            "Method_A_letter": first_label,
            "Method_B": method_b,
            "Method_B_letter": second_label,
            "Winner_Method": winner_method
        }]).to_csv(
            "method_comparisons.csv",
            mode="a",
            index=False,
            header=not os.path.exists("method_comparisons.csv")
        )
        return winning_label, first_label, second_label

    print("Failed comparison")
    return None, first_label, second_label

def pairwise(data, comparisons):

    query = data["Query"].iloc[0]

    letters = [chr(65 + i) for i in range(len(data))]

    docs = dict(zip(letters, data["Verse"]))

    wins = {letter: 0 for letter in letters}

    methods = dict(zip(
    [chr(65+i) for i in range(len(data))],
    data["Method"]
    ))

    for a, b in combinations(letters, 2):

        # Randomize document order
        if np.random.rand() < 0.5:
            first_letter, second_letter = a, b
        else:
            first_letter, second_letter = b, a

        winner, first_label, second_label = judge_pair(
            query,
            docs[first_letter],
            docs[second_letter],
            methods[first_letter],
            methods[second_letter],
            data["Rank"].iloc[0],
            comparisons
        )
        
        if winner is None:
            continue

        if winner == first_label:
            wins[first_letter] += 1

        elif winner == second_label:
            wins[second_letter] += 1

    ranking = sorted(
        wins.keys(),
        key=lambda x: (-wins[x], x)
    )

    print("Wins:", wins)
    print("Ranking:", ranking)

    return ranking

def ranking_to_scores(ranking):

    n = len(ranking)

    return {
        letter: n - idx - 1
        for idx, letter in enumerate(ranking)
    }

def main():
    results=pd.read_csv(INPUT_CSV)

    results=results.loc[
        :,
        ~results.columns.str.contains("^Unnamed")
    ]

    if "Scored" not in results.columns:
        results["Scored"]=False

    ranking_counts = Counter()
    
    # Dataframe to record how methods mathc up to eachother and the frequencies around them 
    comparisons = pd.DataFrame(
    columns=[
        'Query',
        'Rank',
        'Method_A',
        'Method_A_letter',
        'Method_B',
        'Method_B_letter',
        'Winner_Method'
        ]
    )
    
    for i in range(200):

        data=get_results(results)

        if data is None:
            print("Finished")
            break

        ranking=pairwise(data, comparisons)
        
        ranking_str = "".join(ranking)
        ranking_counts[ranking_str] += 1

        scores=ranking_to_scores(ranking)

        letters=[chr(65+i) for i in range(len(data))]

        scored=data.copy()

        scored["Letter"]=letters
        scored["Score"]=scored["Letter"].map(scores)

        results.loc[data.index,"Scored"]=True

        results.to_csv(
            INPUT_CSV,
            index=False
        )

        scored.to_csv(
            OUTPUT_CSV,
            mode="a",
            index=False,
            header=not os.path.exists(OUTPUT_CSV)
        )

        print(f"Saved {OUTPUT_CSV}")

    print("\nRanking Frequencies")
    print("-" * 40)

    total = sum(ranking_counts.values())

    for ranking, count in ranking_counts.most_common():
        pct = count / total * 100
        print(f"{ranking}: {count} ({pct:.2f}%)")
    
    print("Remaining rows:", (results["Scored"] == False).sum())
    
        
    print("Finished scoring.")


if __name__=="__main__":
    main()