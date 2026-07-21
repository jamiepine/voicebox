"""Real-model evaluation for language-aware transcript refinement.

This is deliberately an executable evaluation harness rather than a pytest test:
Qwen output is non-deterministic and failures need human inspection.

Usage:
    python backend/tests/evaluate_multilingual_refinement.py
    python backend/tests/evaluate_multilingual_refinement.py --model 0.6B --quick
    python backend/tests/evaluate_multilingual_refinement.py --json results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.backends.qwen_llm_backend import MLXQwenLLMBackend  # noqa: E402
from backend.services import refinement  # noqa: E402


@dataclass(frozen=True)
class EvalCase:
    language: str
    category: str
    raw: str
    must_contain: tuple[str, ...] = ()
    must_not_contain: tuple[str, ...] = ()
    question: bool = False


CASES: tuple[EvalCase, ...] = (
    EvalCase("en", "question", "uh what time is the deployment in Tokyo on Friday", ("Tokyo", "Friday"), question=True),
    EvalCase("en", "self-correction", "remind me at seven no actually six pm to call mom", ("six",), ("seven",)),
    EvalCase(
        "en", "code-switch", "open package dot json then run the tests on GitHub", ("package.json", "tests", "GitHub")
    ),
    EvalCase(
        "es", "question", "eh a qué hora es el despliegue en Tokio el viernes", ("Tokio", "viernes"), question=True
    ),
    EvalCase(
        "es", "self-correction", "recuérdame a las siete no en realidad a las seis llamar a mamá", ("seis",), ("siete",)
    ),
    EvalCase(
        "es", "code-switch", "abre package dot json y ejecuta los tests en GitHub", ("package.json", "tests", "GitHub")
    ),
    EvalCase(
        "fr", "question", "euh à quelle heure est le déploiement à Tokyo vendredi", ("Tokyo", "vendredi"), question=True
    ),
    EvalCase(
        "fr",
        "self-correction",
        "rappelle-moi à sept heures non en fait à six heures d'appeler maman",
        ("six",),
        ("sept",),
    ),
    EvalCase(
        "fr",
        "code-switch",
        "ouvre package dot json puis lance les tests sur GitHub",
        ("package.json", "tests", "GitHub"),
    ),
    EvalCase("de", "question", "äh wann ist das Deployment in Tokio am Freitag", ("Tokio", "Freitag"), question=True),
    EvalCase(
        "de",
        "self-correction",
        "erinnere mich um sieben nein eigentlich um sechs Mama anzurufen",
        ("sechs",),
        ("sieben",),
    ),
    EvalCase(
        "de",
        "code-switch",
        "öffne package dot json und führe die tests auf GitHub aus",
        ("package.json", "tests", "GitHub"),
    ),
    EvalCase(
        "ja",
        "question",
        "えっと金曜日の東京でのdeploymentは何時ですか",
        ("東京", "金曜日", "deployment"),
        question=True,
    ),
    EvalCase("ja", "self-correction", "母に電話するのを7時いや6時にリマインドして", ("6時",), ("7時",)),
    EvalCase(
        "ja", "code-switch", "package dot jsonを開いてGitHubでtestsを実行して", ("package.json", "GitHub", "tests")
    ),
    EvalCase("zh", "question", "嗯周五在东京的deployment是几点", ("周五", "东京", "deployment"), question=True),
    EvalCase("zh", "self-correction", "提醒我七点不对六点给妈妈打电话", ("六点",), ("七点",)),
    EvalCase("zh", "code-switch", "打开package dot json然后在GitHub运行tests", ("package.json", "GitHub", "tests")),
    EvalCase(
        "hi", "question", "उम शुक्रवार को टोक्यो में deployment कितने बजे है", ("शुक्रवार", "टोक्यो", "deployment"), question=True
    ),
    EvalCase("hi", "self-correction", "मुझे सात बजे नहीं असल में छह बजे माँ को फ़ोन करने की याद दिलाना", ("छह",), ("सात",)),
    EvalCase("hi", "code-switch", "package dot json खोलो और GitHub पर tests चलाओ", ("package.json", "GitHub", "tests")),
)

SCRIPT_PATTERNS = {
    "ja": re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]"),
    "zh": re.compile(r"[\u4e00-\u9fff]"),
    "hi": re.compile(r"[\u0900-\u097f]"),
}


@dataclass
class EvalResult:
    model: str
    language: str
    category: str
    raw: str
    output: str
    passed: bool
    failures: list[str]


def score(case: EvalCase, output: str, model: str) -> EvalResult:
    folded = output.casefold()
    failures = [f"missing {token!r}" for token in case.must_contain if token.casefold() not in folded]
    failures.extend(
        f"retained retracted token {token!r}" for token in case.must_not_contain if token.casefold() in folded
    )
    japanese_question = case.language == "ja" and output.rstrip().endswith("か。")
    if case.question and not japanese_question and not output.rstrip().endswith(("?", "？")):
        failures.append("question did not remain a question")
    script = SCRIPT_PATTERNS.get(case.language)
    if script is not None and script.search(output) is None:
        failures.append("source script was not preserved")
    if not output.strip():
        failures.append("empty output")
    return EvalResult(
        model=model,
        language=case.language,
        category=case.category,
        raw=case.raw,
        output=output,
        passed=not failures,
        failures=failures,
    )


async def run(models: list[str], quick: bool, category: str | None) -> list[EvalResult]:
    backend = MLXQwenLLMBackend(models[0])
    original_getter = refinement.llm_service.get_llm_model
    refinement.llm_service.get_llm_model = lambda: backend
    cases = [
        case
        for case in CASES
        if (not quick or case.category == "code-switch") and (category is None or case.category == category)
    ]
    results: list[EvalResult] = []
    try:
        for model in models:
            for case in cases:
                output, _ = await refinement.refine_transcript(
                    case.raw,
                    refinement.RefinementFlags(),
                    model_size=model,
                    language=case.language,
                )
                result = score(case, output, model)
                results.append(result)
                mark = "PASS" if result.passed else "FAIL"
                print(f"[{mark}] {model:4} {case.language}/{case.category}: {output}")
                for failure in result.failures:
                    print(f"       - {failure}")
    finally:
        refinement.llm_service.get_llm_model = original_getter
        backend.unload_model()
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", choices=("0.6B", "4B"))
    parser.add_argument("--quick", action="store_true", help="Run code-switch cases only")
    parser.add_argument("--category", choices=("question", "self-correction", "code-switch"))
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    models = args.model or ["0.6B", "4B"]
    results = asyncio.run(run(models, args.quick, args.category))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2) + "\n")
    failures = sum(not result.passed for result in results)
    print(f"\n{len(results) - failures}/{len(results)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
