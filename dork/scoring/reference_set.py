from __future__ import annotations

import json
import logging
from pathlib import Path

from dork.scoring.embeddings import fetch_embedding

log = logging.getLogger(__name__)

DEFAULT_PATH = Path("data/reference_set.jsonl")

# Landmark AI engineering papers for seeding the reference set
SEED_PAPERS = [
    ("2201.11903", "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"),
    ("2203.11171", "Self-Consistency Improves Chain of Thought Reasoning in Language Models"),
    ("2210.03629", "ReAct: Synergizing Reasoning and Acting in Language Models"),
    ("2302.13971", "LLaMA: Open and Efficient Foundation Language Models"),
    ("2303.08774", "GPT-4 Technical Report"),
    ("2305.10601", "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"),
    ("2305.18290", "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"),
    ("2307.09288", "Llama 2: Open Foundation and Fine-Tuned Chat Models"),
    ("2312.10997", "Mixtral of Experts"),
    ("2005.14165", "Language Models are Few-Shot Learners"),
    ("2204.02311", "PaLM: Scaling Language Modeling with Pathways"),
    ("2112.11446", "WebGPT: Browser-assisted question-answering with human feedback"),
    ("2203.02155", "Training language models to follow instructions with human feedback"),
    ("2304.03442", "Generative Agents: Interactive Simulacra of Human Behavior"),
    ("2310.06825", "Mistral 7B"),
    ("2205.01068", "OPT: Open Pre-trained Transformer Language Models"),
    ("2305.14314", "QLoRA: Efficient Finetuning of Quantized Language Models"),
    ("2106.09685", "LoRA: Low-Rank Adaptation of Large Language Models"),
    ("2305.11206", "Gorilla: Large Language Model Connected with Massive APIs"),
    ("2310.12931", "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"),
    ("2312.06648", "Dense X Retrieval: What Retrieval Granularity Should We Use?"),
    ("2005.11401", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"),
    ("2208.03299", "Atlas: Few-shot Learning with Retrieval Augmented Language Models"),
    ("2301.12652", "REPLUG: Retrieval-Augmented Black-Box Language Models"),
    ("2212.10560", "Self-Instruct: Aligning Language Models with Self-Generated Instructions"),
    ("2304.12244", "WizardLM: Empowering Large Language Models to Follow Complex Instructions"),
    ("2305.06983", "Voyager: An Open-Ended Embodied Agent with Large Language Models"),
    ("2308.12950", "CodeLlama: Open Foundation Models for Code"),
    ("2310.16944", "Llemma: An Open Language Model For Mathematics"),
    ("2309.16609", "Textbooks Are All You Need II: phi-1.5 technical report"),
    ("2310.01377", "Improved Baselines with Visual Instruction Tuning"),
    ("2306.08568", "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"),
    ("2304.01373", "Instruction Tuning with GPT-4"),
    ("2305.13048", "LIMA: Less Is More for Alignment"),
    ("2309.10305", "Agents: An Open-source Framework for Autonomous Language Agents"),
    ("2310.11511", "OpenAgents: An Open Platform for Language Agents in the Wild"),
    ("2306.05685", "Orca: Progressive Learning from Complex Explanation Traces of GPT-4"),
    ("2306.13549", "Extending Context Window of Large Language Models via Positional Interpolation"),
    ("2309.12307", "Effective Long-Context Scaling of Foundation Models"),
    ("2307.03172", "Lost in the Middle: How Language Models Use Long Contexts"),
    ("2306.02707", "GGML: Large Language Models on Consumer Hardware"),
    ("2210.11416", "Scaling Instruction-Finetuned Language Models"),
    ("2308.10792", "Shepherd: A Critic for Language Model Generation"),
    ("2310.02226", "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines"),
    ("2212.08073", "Constitutional AI: Harmlessness from AI Feedback"),
    ("2305.20050", "Scaling Data-Constrained Language Models"),
    ("2401.02954", "Mixtral of Experts"),
    ("2402.13228", "Gemma: Open Models Based on Gemini Research and Technology"),
    ("2403.08295", "Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking"),
]


class ReferenceSet:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_PATH
        self._entries: list[dict] | None = None

    def load(self) -> list[dict]:
        """Load reference set from JSONL file."""
        if self._entries is not None:
            return self._entries

        self._entries = []
        if not self.path.exists():
            return self._entries

        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self._entries.append(json.loads(line))

        log.info("loaded reference set", extra={"count": len(self._entries)})
        return self._entries

    @property
    def embeddings(self) -> list[list[float]]:
        """Return all reference embeddings."""
        return [e["embedding"] for e in self.load() if e.get("embedding")]

    def seed(self) -> None:
        """Seed the reference set by fetching embeddings from Semantic Scholar."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

        existing_ids = {e["arxiv_id"] for e in self.load()}
        new_count = 0

        with open(self.path, "a") as f:
            for arxiv_id, title in SEED_PAPERS:
                if arxiv_id in existing_ids:
                    continue

                log.info("fetching reference embedding", extra={"arxiv_id": arxiv_id})
                embedding = fetch_embedding(arxiv_id)
                if embedding is None:
                    log.warning("no embedding for reference paper", extra={"arxiv_id": arxiv_id})
                    continue

                entry = {
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "embedding": embedding,
                }
                f.write(json.dumps(entry) + "\n")
                new_count += 1

        log.info("seeded reference set", extra={"new": new_count})
        # Reset cache so next load picks up new entries
        self._entries = None
