"""
NanoVector Agent Episodic Memory Example
Demonstrates how an LLM agent uses NanoVector for instant long-term memory retrieval.
"""

import os
import nanovector
import numpy as np

class AgentMemory:
    def __init__(self, filepath="agent_brain.nvec", dim=384):
        self.filepath = filepath
        self.dim = dim
        if os.path.exists(filepath):
            self.index = nanovector.load(filepath)
            print(f"[Memory] Restored existing brain with {len(self.index)} memories.")
        else:
            self.index = nanovector.Index(dim=dim, metric="cosine")
            print("[Memory] Created fresh agent episodic memory.")

    def store_fact(self, fact_id: str, embedding: np.ndarray, text: str):
        self.index.add(fact_id, embedding, metadata=text)
        self.index.save(self.filepath)
        print(f"[Store] Saved: '{text}' (ID: {fact_id})")

    def recall(self, query_embedding: np.ndarray, top_k=2):
        return self.index.search(query_embedding, top_k=top_k)

def main():
    brain_file = "temp_agent_brain.nvec"
    memory = AgentMemory(filepath=brain_file, dim=384)

    rng = np.random.default_rng(101)

    # Simulate embeddings for 4 facts
    embeddings = {
        "user_pref": rng.standard_normal(384).astype(np.float32),
        "api_key_info": rng.standard_normal(384).astype(np.float32),
        "favorite_food": rng.standard_normal(384).astype(np.float32),
        "project_goal": rng.standard_normal(384).astype(np.float32),
    }

    memory.store_fact("mem_1", embeddings["user_pref"], "User prefers Python and C over Java.")
    memory.store_fact("mem_2", embeddings["api_key_info"], "OpenAI API key is stored in .env.")
    memory.store_fact("mem_3", embeddings["favorite_food"], "User loves Italian espresso and pasta.")
    memory.store_fact("mem_4", embeddings["project_goal"], "Build the fastest vector database on PyPI.")

    # Query: Agent asks "What does the user prefer?"
    # Simulate a query close to user_pref embedding with a little noise
    query_vector = embeddings["user_pref"] + rng.normal(0, 0.1, 384).astype(np.float32)

    print("\n[Query] Agent searching for user preferences...")
    matches = memory.recall(query_vector, top_k=2)

    for i, match in enumerate(matches, 1):
        print(f"  Result #{i}: Score {match.score:.4f} => {match.metadata}")

    if os.path.exists(brain_file):
        os.remove(brain_file)

if __name__ == "__main__":
    main()
