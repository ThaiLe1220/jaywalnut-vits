import re
from text.japanese import japanese_to_romaji_with_accent
from text.english import english_to_ipa2


# def japanese_cleaners(text):
#     text = f"[JA]{text}[JA]"
#     text = re.sub(
#         r"\[JA\](.*?)\[JA\]",
#         lambda x: japanese_to_romaji_with_accent(x.group(1))
#         .replace("ts", "ʦ")
#         .replace("u", "ɯ")
#         .replace("...", "…")
#         + " ",
#         text,
#     )
#     text = re.sub(r"\s+$", "", text)
#     text = re.sub(r"([^\.,!\?\-…~])$", r"\1.", text)
#     return text


# Assuming imports are correctly set up as per your provided snippets.


def english_cleaners(text):
    text = f"[EN]{text}[EN]"  # Mark the text for processing, similar to the Japanese cleaner.

    # Replace the English text enclosed in markers with its IPA2 representation.
    # Additional replacements and formatting can be applied here as needed.
    text = re.sub(
        r"\[EN\](.*?)\[EN\]",
        lambda x: english_to_ipa2(x.group(1))
        .replace("...", "…")  # Normalize ellipsis
        .strip()  # Ensure no leading/trailing whitespace
        + " ",  # Add a space at the end for consistency
        text,
    )

    # Normalize multiple spaces to a single space.
    text = re.sub(r"\s{2,}", " ", text)

    # Clean up: remove any excess whitespace at the end of the string.
    text = re.sub(r"\s+$", "", text)
    text = re.sub(r"([?!.])", r"\1 ", text)  # Ensure space after punctuation.

    # Ensure punctuation: if the text doesn't end with common punctuation, add a period.
    # This mirrors the Japanese cleaner functionality for consistency.
    text = re.sub(r"([^\.,!\?\-…~])$", r"\1.", text)

    return text


""" from https://github.com/keithito/tacotron """

"""
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
"""

import re
from unidecode import unidecode
from phonemizer import phonemize


# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    """Pipeline for non-English text that transliterates to ASCII."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners2(text):
    """Pipeline for English text, including abbreviation expansion. + punctuation + stress"""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = phonemize(
        text,
        language="en-us",
        backend="espeak",
        strip=True,
        preserve_punctuation=True,
        with_stress=True,
    )
    phonemes = collapse_whitespace(phonemes)
    return phonemes
