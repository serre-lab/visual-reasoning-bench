#!/usr/bin/env python
"""Simple script to exercise the ChatGPT vision model wrapper."""

import argparse
import sys
from pathlib import Path

# Add bench/ to PYTHONPATH so we can import the package when called directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from bench.models import ChatGPTVisionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single ChatGPT vision prompt against a local image."
    )
    parser.add_argument(
        "--image",
        type=str,
        default="assets/klimt.jpg",
        help="Path to the image file to send to ChatGPT.",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Is this a famous painter?",
        help="Question to pair with the image.",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-4o-mini",
        help="Vision-capable OpenAI model identifier.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=200,
        help="Maximum number of output tokens to request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Decoding temperature (0 provides deterministic results).",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Optional system prompt / instructions for the assistant.",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="Override the OpenAI API base URL (for Azure proxies, etc.).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = ChatGPTVisionModel(
        openai_model=args.openai_model,
        api_base=args.api_base,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        system_prompt=args.system_prompt,
    )
    prediction = model.predict(args.image, args.question)
    print("\nPrediction")
    print("-" * 40)
    print(prediction)


if __name__ == "__main__":
    main()
