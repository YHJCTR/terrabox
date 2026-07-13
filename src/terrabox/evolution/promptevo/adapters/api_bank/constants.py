"""Static prompt constants and parse constants for API-Bank."""
from __future__ import annotations

import re

API_BANK_API_CALL_PROMPT = """Based on the given API description and the existing conversation history 1..t, please generate the API request that the AI should call in step t+1 and output it in the format of [ApiName(key1='value1', key2='value2', ...)], replace the ApiName with the actual API name, and replace the key and value with the actual parameters.
Your output should start with a square bracket "[" and end with a square bracket "]". Do not output any other explanation or prompt or the result of the API call in your output.
This year is 2023.
Input:
User: [User's utterence]
AI: [AI's utterence]

Expected output:
[ApiName(key1='value1', key2='value2', ...)]

API descriptions:
"""

API_BANK_RESPONSE_PROMPT = """Based on the given API description and the existing conversation history 1..t, please generate the next dialog that the AI should response after the API call t.
This year is 2023.
Input:
User: [User's utterence]
AI: [AI's utterence]
[ApiName(key1='value1', key2='value2', ...)]

Expected output:
AI: [AI's utterence]

API descriptions:
"""

# Backward-compatible name: this is the static task instruction slot optimized
# by promptevo. Dynamic API descriptions and dialogue history are appended later.
API_BANK_SYSTEM_PROMPT = API_BANK_API_CALL_PROMPT


_API_CALL_RE = re.compile(r"\[(\w+)\((.*)\)\]", re.MULTILINE | re.DOTALL)
_OFFICIAL_ERROR_CODES = (
    "NO_API_CALL",
    "API_NAME_MISMATCH",
    "HAS_EXCEPTION",
    "INPUT_MISMATCH",
    "OUTPUT_MISMATCH",
    "INVALID_INPUT_PARAMETER",
    "KEY_ERROR",
    "FAILED_PARSE_API_CALL",
    "MISS_INPUT_ARGUMENT",
)
