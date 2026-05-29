"""Per-passage symbolic featurizer (psi).

Each retrieved passage is mapped to a tuple of discrete attributes:

  stance                  query-conditioned NLI label: {entail, neutral, contradict}
  entity_overlap          Jaccard of NER entities between query and passage, bucketed
  lex_overlap             query alphabetic token coverage in the passage, bucketed
  length                  passage length bucket: {short, med, long}
  has_numeric             1 if the passage contains a numeric quantity (regex)
  has_date                1 if the passage contains a year/date (regex)
  source                  dataset-driven source class (heuristic)
  redundant_with_sibling  1 if cosine-similar to another passage in this set

These attributes are intentionally coarse and human-readable. The state-
discovery objective is to find recurring *configurations* over the multiset
of these tuples, not to fit raw text embeddings.
"""
import re

import numpy as np
import spacy
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from ..utils.logging import get_logger

LOG = get_logger()

# --- regex sets ---
NUM_RX = re.compile(
    r"(?<!\w)[-+]?\d[\d,]*(?:\.\d+)?(?:\s?(?:%|kg|g|mg|cm|mm|m|km|s|ms|°C|°F|usd|eur|gbp))?(?!\w)",
    re.IGNORECASE,
)
DATE_RX = re.compile(
    r"\b(?:19|20)\d{2}\b"
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,\s*\d{2,4})?\b"
    r"|\b\d{1,2}/\d{1,2}/(?:\d{2}|\d{4})\b",
    re.IGNORECASE,
)


class FeatureExtractor:
    def __init__(
        self,
        nli_model="cross-encoder/nli-deberta-v3-small",
        embed_model="sentence-transformers/all-MiniLM-L6-v2",
        spacy_model="en_core_web_sm",
        device="auto",
        short_max=200,
        long_min=800,
        redundancy_threshold=0.85,
        batch_nli=32,
        batch_embed=64,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        LOG.info(f"Loading NLI ({nli_model}) on {device}")
        self.nli = CrossEncoder(nli_model, device=device)
        LOG.info(f"Loading embedder ({embed_model}) on {device}")
        self.embedder = SentenceTransformer(embed_model, device=device)
        LOG.info(f"Loading spaCy ({spacy_model})")
        self.nlp = spacy.load(spacy_model, disable=["parser", "lemmatizer"])
        self.short_max = short_max
        self.long_min = long_min
        self.redundancy_threshold = redundancy_threshold
        self.batch_nli = batch_nli
        self.batch_embed = batch_embed
        # NLI label index -> {entail, neutral, contradict}
        try:
            id2lab = self.nli.model.config.id2label
            self.label_map = {}
            for i, lab in id2lab.items():
                low = str(lab).lower()
                if "entail" in low:
                    self.label_map[int(i)] = "entail"
                elif "contradict" in low:
                    self.label_map[int(i)] = "contradict"
                else:
                    self.label_map[int(i)] = "neutral"
            LOG.info(f"NLI label map: {self.label_map}")
        except Exception:
            self.label_map = {0: "contradict", 1: "neutral", 2: "entail"}

    # ---- attribute computations ----

    def _stance(self, query, texts):
        if not texts:
            return []
        pairs = [(t, query) for t in texts]  # premise=passage, hypothesis=query
        scores = self.nli.predict(
            pairs,
            batch_size=self.batch_nli,
            show_progress_bar=False,
            apply_softmax=True,
            convert_to_numpy=True,
        )
        if scores.ndim == 1:
            scores = scores.reshape(1, -1)
        out = []
        for s in scores:
            i = int(np.argmax(s))
            out.append(self.label_map.get(i, "neutral"))
        return out

    @staticmethod
    def _entity_overlap(q_ents, p_ents):
        a = {e.lower() for e in q_ents}
        b = {e.lower() for e in p_ents}
        if not a:
            return "low"
        denom = len(a | b)
        if denom == 0:
            return "low"
        j = len(a & b) / denom
        if j < 0.05:
            return "low"
        if j < 0.30:
            return "med"
        return "high"

    @staticmethod
    def _lex_overlap(q_tokens, p_tokens):
        a = {t.lower() for t in q_tokens if t.isalpha()}
        b = {t.lower() for t in p_tokens if t.isalpha()}
        if not a:
            return "low"
        j = len(a & b) / len(a)
        if j < 0.15:
            return "low"
        if j < 0.50:
            return "med"
        return "high"

    def _length_bucket(self, text):
        n = len(text)
        if n < self.short_max:
            return "short"
        if n > self.long_min:
            return "long"
        return "med"

    @staticmethod
    def _has_numeric(text):
        return 1 if NUM_RX.search(text or "") else 0

    @staticmethod
    def _has_date(text):
        return 1 if DATE_RX.search(text or "") else 0

    @staticmethod
    def _source_class(dataset_name, _passage_meta):
        dn = (dataset_name or "").lower()
        if "fiqa" in dn:
            return "finance_web"
        if "nfcorpus" in dn:
            return "biomed"
        if "scifact" in dn:
            return "scientific"
        if "hotpot" in dn:
            return "wiki"
        return "other"

    # ---- top-level ----

    def featurize_set(self, query_text, passages, dataset_name="?"):
        """passages: list of {'doc_id','text','title',...}. Returns list of attr dicts."""
        if not passages:
            return []
        texts = [p.get("text", "") or "" for p in passages]
        stances = self._stance(query_text, texts)
        embs = self.embedder.encode(
            texts,
            batch_size=self.batch_embed,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        n = len(texts)
        sim = embs @ embs.T
        np.fill_diagonal(sim, 0.0)
        max_sib = sim.max(axis=1) if n > 1 else np.zeros(n)
        redundancy = (max_sib > self.redundancy_threshold).astype(int)

        docs = list(self.nlp.pipe([query_text] + texts))
        q_doc = docs[0]
        q_ents = [e.text for e in q_doc.ents]
        q_tokens = [t.text for t in q_doc]
        out = []
        for i, t in enumerate(texts):
            p_doc = docs[i + 1]
            p_ents = [e.text for e in p_doc.ents]
            p_tokens = [tok.text for tok in p_doc]
            out.append(
                {
                    "stance": stances[i] if i < len(stances) else "neutral",
                    "entity_overlap": self._entity_overlap(q_ents, p_ents),
                    "lex_overlap": self._lex_overlap(q_tokens, p_tokens),
                    "length": self._length_bucket(t),
                    "has_numeric": int(self._has_numeric(t)),
                    "has_date": int(self._has_date(t)),
                    "source": self._source_class(dataset_name, passages[i]),
                    "redundant_with_sibling": int(redundancy[i]),
                }
            )
        return out
