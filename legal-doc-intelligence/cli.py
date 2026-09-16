"""Command-line interface for LegalLens."""
import argparse
import json
import sys
sys.path.insert(0, ".")
from src.pipeline import process_document


def cmd_summarize(args):
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()
    result = process_document(
        text,
        redact_names=args.redact_names,
        strip_preamble=not args.no_strip_preamble,
    )

    if args.mode in ("lawyer", "both"):
        print("=" * 70)
        print("LAWYER MODE")
        print("=" * 70)
        print(result["lawyer_mode"]["full_text"])
        print(f"\n[Word count: {result['lawyer_mode']['word_count']}]")
        print(f"[Verification: {result['lawyer_verification']}]")
        print()

    if args.mode in ("citizen", "both"):
        print("=" * 70)
        print("CITIZEN MODE")
        print("=" * 70)
        print(result["citizen_mode"]["full_text"])
        print(f"\n[Word count: {result['citizen_mode']['word_count']}]")
        print(f"[Verification: {result['citizen_verification']}]")
        if result["citizen_mode"].get("glossary"):
            print("\nGlossary:")
            for term, definition in result["citizen_mode"]["glossary"].items():
                print(f"  {term}: {definition}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n[Full result written to {args.output}]")


def cmd_inspect(args):
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    from src.data.cleaning import preprocess, split_paragraphs
    from src.data.section_detector import detect_sections, group_by_section

    clean = preprocess(text)
    paragraphs = split_paragraphs(clean)
    para_objects = detect_sections(paragraphs)
    groups = group_by_section(para_objects)

    print(f"Total paragraphs: {len(paragraphs)}")
    print(f"Sections detected: {list(groups.keys())}")
    print()
    from collections import Counter
    counts = Counter(p.section for p in para_objects)
    for section, count in counts.most_common():
        pct = count / len(para_objects) * 100
        print(f"  {section:25s} {count:4d} paragraphs ({pct:.1f}%)")


def cmd_sample(args):
    """Generate a sample judgment text file for testing."""
    sample = """The Judgment of the Court was delivered by SAHAI, J.

The appellant was employed as an Assistant Engineer in the Central Public
Works Department. He was placed under suspension on September 3, 1959,
pending a departmental inquiry.

The appellant filed a writ petition in the High Court challenging the
prolonged suspension. Learned counsel for the appellant contended that the
suspension was unjustified and lasted over ten years without any decision.

The respondent State submitted that the suspension was necessary during the
pendency of the departmental inquiry, and that the appellant had no vested
right to be reinstated.

We have considered the arguments of both sides. In our opinion, a suspension
that continues for over a decade without a determination violates principles
of natural justice. The Court observes that Fundamental Rule 54 requires the
government to pass an order regarding the period of suspension.

The order of compulsory retirement dated 25.4.1972 is set aside. The appeal
is allowed with costs.
"""
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(sample)
    print(f"Wrote sample judgment to {args.out}")


def main():
    parser = argparse.ArgumentParser(description="LegalLens CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sum = sub.add_parser("summarize", help="Generate summaries")
    p_sum.add_argument("input", help="Path to judgment text file")
    p_sum.add_argument("--mode", choices=["lawyer", "citizen", "both"], default="both")
    p_sum.add_argument("--output", help="Optional JSON output path")
    p_sum.add_argument("--redact-names", action="store_true")
    p_sum.add_argument("--no-strip-preamble", action="store_true")
    p_sum.set_defaults(func=cmd_summarize)

    p_ins = sub.add_parser("inspect", help="Inspect document structure")
    p_ins.add_argument("input", help="Path to judgment text file")
    p_ins.set_defaults(func=cmd_inspect)

    p_smp = sub.add_parser("sample", help="Write a sample judgment for testing")
    p_smp.add_argument("--out", default="judgment.txt")
    p_smp.set_defaults(func=cmd_sample)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
