#!/usr/bin/env python3

import re
import spacy
from pathlib import Path


UMLS_MEDICAL_SEMTYPES = {
    "T017", "T022", "T023", "T024", "T025", "T029", "T030",
    "T103", "T109", "T116", "T121", "T122", "T123", "T125",
    "T126", "T127", "T129", "T130", "T131", "T195",
    "T020", "T033", "T034", "T037", "T046", "T047", "T048",
    "T049", "T050", "T184", "T190", "T191",
    "T059", "T060", "T061", "T065",
    "T074", "T075",
}


def load_meddra_vocab(meddra_dir: str) -> set:

    vocab = set()
    meddra_path = Path(meddra_dir)

    files_to_load = [
        meddra_path / "MedAscii" / "llt.asc",
        meddra_path / "MedAscii" / "pt.asc",
        meddra_path / "llt.asc",
        meddra_path / "pt.asc",
    ]

    loaded = False
    for fpath in files_to_load:
        if not fpath.exists():
            continue
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.strip().split("$")
                if len(parts) < 2:
                    continue
                term = parts[1].strip().lower()
                if term:
                    vocab.add(term)
                    for word in term.split():
                        word = re.sub(r"[^a-z0-9\-]", "", word)
                        if len(word) >= 3:
                            vocab.add(word)
        print(f"  Loaded MedDRA vocab from {fpath.name}: {len(vocab)} terms")
        loaded = True

    if not loaded:
        print(f"  Warning: MedDRA files not found in {meddra_dir}. Skipping MedDRA vocab.")

    return vocab


def load_umls_via_scispacy(nlp) -> set:

    vocab = set()
    try:
        from scispacy.linking import EntityLinker
        if "scispacy_linker" not in nlp.pipe_names:
            nlp.add_pipe(
                "scispacy_linker",
                config={"resolve_abbreviations": True, "linker_name": "umls"},
                last=True
            )
        linker = nlp.get_pipe("scispacy_linker")
        kb = linker.kb

        for cui, entity in kb.cui_to_entity.items():
            if not any(st in UMLS_MEDICAL_SEMTYPES for st in entity.types):
                continue
            vocab.add(entity.canonical_name.lower())
            for alias in entity.aliases:
                a = alias.lower().strip()
                if a:
                    vocab.add(a)
                    for word in a.split():
                        word = re.sub(r"[^a-z0-9\-]", "", word)
                        if len(word) >= 3:
                            vocab.add(word)

        print(f"  Loaded UMLS vocab via scispaCy linker: {len(vocab)} terms")
    except ImportError:
        print("  Warning: scispacy[umls] not installed. Run: pip install scispacy[umls]")
    except Exception as e:
        print(f"  Warning: UMLS linker load failed: {e}")

    return vocab


class MedicalTermDetector:


    MEDICAL_ENTITY_TYPES = {
        "CHEMICAL", "DISEASE", "GENE_OR_GENE_PRODUCT",
        "CANCER", "CELL_TYPE", "CELL_LINE", "DNA", "RNA",
        "SIMPLE_CHEMICAL", "ORGANISM", "CELLULAR_COMPONENT",
        "IMMATERIAL_ANATOMICAL_ENTITY", "MULTI-TISSUE_STRUCTURE",
        "ORGAN", "ORGANISM_SUBDIVISION", "ORGANISM_SUBSTANCE",
        "PATHOLOGICAL_FORMATION", "TISSUE",
    }

    def __init__(
        self,
        meddra_dir: str = None,
        use_umls_linker: bool = False,
        spacy_model: str = None,
    ):
        self._vocab: set = set()
        self.nlp = None

        model_candidates = (
            [spacy_model] if spacy_model
            else ["en_ner_bc5cdr_md", "en_core_sci_lg", "en_core_sci_sm", "en_core_web_sm"]
        )
        for name in model_candidates:
            try:
                self.nlp = spacy.load(name, disable=["parser", "textcat"])
                print(f"  Loaded spaCy model: {name}")
                break
            except OSError:
                continue
        if self.nlp is None:
            raise RuntimeError(
                "No spaCy model found. Install one of:\n"
                "  pip install scispacy\n"
                "  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy"
                "/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz"
            )

        if use_umls_linker:
            self._vocab.update(load_umls_via_scispacy(self.nlp))

        if meddra_dir:
            self._vocab.update(load_meddra_vocab(meddra_dir))

        if not self._vocab:
            print("  No external vocab loaded. Using scispaCy NER only.")

        print(f"  MedicalTermDetector ready. Vocab size: {len(self._vocab)}")

    def analyze_document(self, text: str) -> dict:

        results = {}
        doc = self.nlp(text)

        ner_tokens = set()
        ner_types = {}
        for ent in doc.ents:
            if ent.label_ in self.MEDICAL_ENTITY_TYPES or len(self._vocab) == 0:
                for token in ent:
                    t = token.text.lower().strip()
                    ner_tokens.add(t)
                    ner_types[t] = ent.label_

        for token in doc:
            t = token.text.lower().strip()
            if not t or not re.search(r"[a-z]", t):
                continue

            in_ner = t in ner_tokens
            in_vocab = t in self._vocab or token.lemma_.lower() in self._vocab

            if in_ner and in_vocab:
                source = "ner+vocab"
            elif in_ner:
                source = "ner"
            elif in_vocab:
                source = "vocab"
            else:
                continue

            results[t] = {
                "is_medical": True,
                "source": source,
                "entity_type": ner_types.get(t, "VOCAB"),
            }

        return results
