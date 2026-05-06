"""
Romanian Legislative Ontological Reasoning.
Applies OWL/SWRL-like logical axioms specific to Romanian legislative knowledge graphs.
"""

import pandas as pd

from .config import (
    LEGISLATIVE_ASYMMETRIC_RELATIONS,
    LEGISLATIVE_CONSTRAINTS,
    LEGISLATIVE_FUNCTIONAL_RELATIONS,
    LEGISLATIVE_IRREFLEXIVE_RELATIONS,
    LEGISLATIVE_SYMMETRIC_RELATIONS,
    LEGISLATIVE_TRANSITIVE_RELATIONS,
)


class LegislativeOntologyReasoner:
    """Ontology reasoner for Romanian legislative knowledge graphs."""

    def __init__(self, df):
        self.df = df
        self.total_triples = len(df)
        self._has_provenance = "law_id" in df.columns and "article_number" in df.columns

        print("\n" + "=" * 70)
        print(" ROMANIAN LEGISLATIVE ONTOLOGICAL REASONING")
        print("=" * 70)
        print(f"Analyzing {self.total_triples:,} legislative triples...")

    def _provenance(self, row) -> str:
        """Return ' (lege_X / art Y)' if provenance is available, else ''."""
        if not self._has_provenance:
            return ""
        law = row.get("law_id")
        art = row.get("article_number")
        if pd.isna(law):
            return ""
        return f" ({law} / art {art})" if not pd.isna(art) else f" ({law})"

    def verify_functional_properties(self):
        """Functional: a head can have at most one tail per such relation."""
        print("\n[Axiom 1] Functional Properties")
        print(f"Properties: {LEGISLATIVE_FUNCTIONAL_RELATIONS}")

        subset = self.df[self.df["relation"].isin(LEGISLATIVE_FUNCTIONAL_RELATIONS)]

        if subset.empty:
            print("  ℹ  No functional property triples found.")
            return

        violations = subset.groupby(["head", "relation"])["tail"].nunique().reset_index()
        violations = violations[violations["tail"] > 1]

        if violations.empty:
            print(f"  ✅ Success: All {len(subset)} functional property triples are valid.")
        else:
            print(f"  ❌ Violations: {len(violations)} entities with multiple values:")
            for _, row in violations.head(5).iterrows():
                examples = subset[
                    (subset["head"] == row["head"]) & (subset["relation"] == row["relation"])
                ]
                print(f"     '{row['head']}' has {row['tail']} values for '{row['relation']}':")
                print(f"        {examples['tail'].tolist()}")

    def verify_asymmetric_properties(self):
        """Asymmetric: if (A,r,B) holds, (B,r,A) must not."""
        print("\n[Axiom 2] Asymmetric Properties")
        print(f"Properties: {LEGISLATIVE_ASYMMETRIC_RELATIONS}")

        subset = self.df[self.df["relation"].isin(LEGISLATIVE_ASYMMETRIC_RELATIONS)]

        if subset.empty:
            print("  ℹ  No asymmetric property triples found.")
            return

        edges = set(zip(subset["head"], subset["tail"], subset["relation"]))
        violations = []

        for head, tail, rel in edges:
            if (tail, head, rel) in edges:
                pair = tuple(sorted([head, tail]))
                violations.append((pair[0], pair[1], rel))

        violations = list(set(violations))

        if not violations:
            print(f"  ✅ Success: All {len(subset)} asymmetric triples are valid.")
        else:
            print(f"  ❌ Violations: {len(violations)} circular relationships found:")
            for v in violations[:5]:
                print(f"     '{v[0]}' ↔ '{v[1]}' (via '{v[2]}')")

    def verify_irreflexive_properties(self):
        """Irreflexive: head must differ from tail."""
        print("\n[Axiom 3] Irreflexive Properties (No Self-Loops)")
        print(f"Properties: {LEGISLATIVE_IRREFLEXIVE_RELATIONS}")

        subset = self.df[self.df["relation"].isin(LEGISLATIVE_IRREFLEXIVE_RELATIONS)]

        if subset.empty:
            print("  ℹ  No irreflexive property triples found.")
            return

        violations = subset[subset["head"] == subset["tail"]]

        if violations.empty:
            print(f"  ✅ Success: All {len(subset)} irreflexive triples are valid.")
        else:
            print(f"  ❌ Violations: {len(violations)} self-referential triples:")
            for _, row in violations.head(5).iterrows():
                print(
                    f"     '{row['head']}' --[{row['relation']}]--> itself"
                    f"{self._provenance(row)}"
                )

    def verify_symmetric_properties(self):
        """Symmetric: every (A,r,B) requires its reciprocal (B,r,A)."""
        print("\n[Axiom 4] Symmetric Properties")
        print(f"Properties: {LEGISLATIVE_SYMMETRIC_RELATIONS}")

        subset = self.df[self.df["relation"].isin(LEGISLATIVE_SYMMETRIC_RELATIONS)]

        if subset.empty:
            print("  ℹ  No symmetric property triples found.")
            return

        edges = set(zip(subset["head"], subset["tail"], subset["relation"]))
        missing_reciprocals = []

        for head, tail, rel in edges:
            if (tail, head, rel) not in edges and head != tail:
                missing_reciprocals.append((head, tail, rel))

        if not missing_reciprocals:
            print(f"  ✅ Success: All {len(subset)} symmetric triples have reciprocals.")
        else:
            print(
                f"  [WARN] Incomplete: {len(missing_reciprocals)} missing reciprocal relationships:"
            )
            for mr in missing_reciprocals[:5]:
                print(f"     '{mr[0]}' --[{mr[2]}]--> '{mr[1]}' (missing reverse)")

    def verify_domain_constraints(self):
        """Domain constraints: tail must be in the allowed list per relation."""
        print("\n[Axiom 5] Domain Constraints")

        all_valid = True

        for relation, valid_entities in LEGISLATIVE_CONSTRAINTS.items():
            subset = self.df[self.df["relation"] == relation.replace("_domain", "")]

            if subset.empty:
                continue

            violations = subset[~subset["tail"].isin(valid_entities)]

            if violations.empty:
                print(f"  ✅ '{relation.replace('_domain', '')}': All entities valid")
            else:
                all_valid = False
                print(
                    f"  ❌ '{relation.replace('_domain', '')}': {len(violations)} invalid entities:"
                )
                invalid_entities = violations["tail"].unique()
                for entity in invalid_entities[:3]:
                    print(f"     Invalid: '{entity}' (Expected one of: {valid_entities})")

        if all_valid:
            print("  ✅ All domain constraints satisfied.")

    def verify_transitive_closure(self):
        """Transitive: enumerate 2-hop chains A→B→C for transitive relations."""
        print("\n[Axiom 6] Transitive Property Analysis")
        print(f"Properties: {LEGISLATIVE_TRANSITIVE_RELATIONS}")

        for relation in LEGISLATIVE_TRANSITIVE_RELATIONS:
            subset = self.df[self.df["relation"] == relation]

            if subset.empty:
                continue

            # Build modification chains
            chains = []
            for _, row in subset.iterrows():
                # Find if tail modifies something else
                next_level = subset[subset["head"] == row["tail"]]
                if not next_level.empty:
                    for _, next_row in next_level.iterrows():
                        chains.append((row["head"], row["tail"], next_row["tail"]))

            if chains:
                print(f"  [INFO] Found {len(chains)} transitive chains for '{relation}':")
                for chain in chains[:3]:
                    print(f"     '{chain[0]}' → '{chain[1]}' → '{chain[2]}'")
            else:
                print(f"  ℹ  No transitive chains found for '{relation}'")

    def run_all_tests(self):
        """Run all ontology axiom tests."""
        self.verify_functional_properties()
        self.verify_asymmetric_properties()
        self.verify_irreflexive_properties()
        self.verify_symmetric_properties()
        self.verify_domain_constraints()
        self.verify_transitive_closure()

        print("\n" + "=" * 70)
        print(" ONTOLOGICAL REASONING COMPLETE")
        print("=" * 70)
