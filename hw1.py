#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."

def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Suggested imports:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_deepseek import ChatDeepSeek

    Use the vision-capable DeepSeek Flash model named
    ``deepseek-v4-flash-vision-exp``. The API key is loaded from .env.
    """
    ### YOUR CODE HERE
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.runnables import RunnableLambda
    from langchain_deepseek import ChatDeepSeek
    import logging

    logger = logging.getLogger(__name__)

    # Safe Defaults on Failure to Prevent Program Crashes
    FAILED_RESULT = {
        "amount_paid_after_rounding": 0.0,
        "subtotal_after_discounts_before_rounding": 0.0,
        "discount_total": 0.0,
        "amount_without_discounts": 0.0,
    }

    llm = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        temperature=0,
    )

    prompt_parse = ChatPromptTemplate.from_messages([
        ("system", """You are an expert receipt parser. Read the receipt image and extract EXACTLY THREE numbers.

1) amount_paid_after_rounding
   - The final amount the customer actually paid (the last payable line on the receipt).
   - This is the value AFTER any rounding adjustment.
   - Common labels: "Amount Paid", "Total Paid", "Net Total", "应付款".

2) subtotal_after_discounts_before_rounding
   - The subtotal AFTER discounts/coupons have been subtracted, but BEFORE any rounding adjustment.
   - Common labels: "Subtotal", "Net", "Sub-total after discount", "折后小计".
   - If the receipt only shows a single "Subtotal" plus discount lines, this is (raw subtotal - sum of discounts).

3) discount_total
   - The SUM of all discount / coupon / savings amounts, taken as POSITIVE numbers.
   - If there are no discounts, use 0.

CRITICAL RULES:
- Do NOT output amount_without_discounts; the caller will compute it.
- Do NOT include currency symbols (HK$, $) or thousands separators (,).
- Output numbers as plain decimals (e.g. 102.31, not "102.31" with commas).
- Output ONLY a valid JSON object. No markdown fences, no commentary, no leading/trailing text.

Required JSON schema:
{{"amount_paid_after_rounding": <float>, "subtotal_after_discounts_before_rounding": <float>, "discount_total": <float>}}

Example output:
{{"amount_paid_after_rounding": 102.30, "subtotal_after_discounts_before_rounding": 102.31, "discount_total": 5.39}}"""),
        ("human", [
            {"type": "text", "text": "Extract the three amounts from this receipt image."},
            {"type": "image_url", "image_url": {"url": "{image_data}"}},
        ]),
    ])

    parse_chain = prompt_parse | llm | JsonOutputParser()

    def _safe_finalize(parsed: Any) -> dict:
        """Safely extract fields and compute the fourth value, never letting the chain crash."""
        if not isinstance(parsed, dict):
            logger.warning("Parse returned non-dict: %r", parsed)
            return dict(FAILED_RESULT)
        try:
            subtotal = float(parsed["subtotal_after_discounts_before_rounding"])
            discount = float(parsed["discount_total"])
            paid = float(parsed["amount_paid_after_rounding"])
            return {
                "amount_paid_after_rounding": paid,
                "subtotal_after_discounts_before_rounding": subtotal,
                "discount_total": discount,
                "amount_without_discounts": round(subtotal + discount, 2),
            }
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("Missing/invalid field: %s | parsed=%r", e, parsed)
            return dict(FAILED_RESULT)

    # Add with_fallbacks as a Fallback for API Network Exceptions
    return parse_chain | RunnableLambda(_safe_finalize).with_fallbacks(
        [RunnableLambda(lambda _: dict(FAILED_RESULT))]
    )



def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """
    # 1. Construct Batch Input　
    inputs = [{"image_data": image_data_url(path)} for path in images]

    # 2. Execute Multiple Image Chains in Parallel (Not One by One)
    results: list[dict[str, Any]] = chain.batch(inputs)

    # 3. Exact Summation Using Python’s Decimal (Without Using a Large Model to Compute)
    total_paid = sum(
        (Decimal(str(r.get("amount_paid_after_rounding", 0.0))) for r in results if r),
        Decimal("0"),
    )
    total_without_discount = sum(
        (Decimal(str(r.get("amount_without_discounts", 0.0))) for r in results if r),
        Decimal("0"),
    )

    return {
        QUERY_1: f"{total_paid:.2f}",
        QUERY_2: f"{total_without_discount:.2f}",
    }

# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
