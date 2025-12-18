#!/usr/bin/env python3
"""Utility script to extract text and raster images from the MaskCrypt paper."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

try:
    from pypdf import PdfReader
except ImportError as exc:  # pragma: no cover - convenience for manual runs
        raise SystemExit(
            "Missing dependency 'pypdf'. Install via `pip install pypdf>=4.2.0`."
        ) from exc


DEFAULT_PDF = Path(__file__).with_name("MaskCrypt_Federated_Learning_With_Selective_Homomorphic_Encryption.pdf")


def extract_pdf_assets(
    pdf_path: Path,
    output_dir: Path,
    *,
    extract_text: bool = True,
    extract_images: bool = True,
) -> tuple[int, int]:
    """Save per-page text and embedded images for later review."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    text_dir = output_dir / "text"
    image_dir = output_dir / "images"
    if extract_text:
        text_dir.mkdir(parents=True, exist_ok=True)
    if extract_images:
        image_dir.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(str(pdf_path))
    text_pages = 0
    image_count = 0
    for page_index, page in enumerate(reader.pages, start=1):
        if extract_text:
            text = page.extract_text() or ""
            target = text_dir / f"page_{page_index:03d}.txt"
            target.write_text(text.strip(), encoding="utf-8")
            text_pages += 1
        if extract_images:
            for image_index, image in enumerate(getattr(page, "images", []) or [], start=1):
                suffix = getattr(image, "image_format", None)
                if not suffix:
                    name_hint = str(getattr(image, "name", "")).split(".")[-1].lower()
                    suffix = name_hint if 1 <= len(name_hint) <= 4 else "bin"
                image_path = image_dir / f"page_{page_index:03d}_{image_index:02d}.{suffix}"
                image_path.write_bytes(image.data)
                image_count += 1
    return text_pages, image_count


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="Path to the PDF to parse.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_name("maskcrypt_extracted"),
        help="Output directory where assets will be written.",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Skip text extraction.",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip image extraction.",
    )
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    pages, images = extract_pdf_assets(
        pdf_path=args.pdf,
        output_dir=args.out,
        extract_text=not args.no_text,
        extract_images=not args.no_images,
    )
    print(f"Saved text for {pages} pages and extracted {images} images into {args.out}")


if __name__ == "__main__":
    main()
