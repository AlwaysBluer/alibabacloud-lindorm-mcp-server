import re
from typing import Iterable


IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
FIELD_PATH_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
INDEX_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,254}$")
LINDORM_INSTANCE_ID_RE = re.compile(r"^ld-[A-Za-z0-9]{8,64}$")
MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
MIN_API_KEY_LENGTH = 32
MAX_QUERY_TEXT_LENGTH = 4096

FORBIDDEN_READONLY_TOKENS = {
    "alter",
    "call",
    "create",
    "delete",
    "drop",
    "execute",
    "grant",
    "insert",
    "load",
    "merge",
    "outfile",
    "replace",
    "revoke",
    "set",
    "truncate",
    "update",
    "use",
}


def _quote_identifier(identifier: str) -> str:
    if not IDENTIFIER_RE.fullmatch(identifier):
        raise ValueError(f"Invalid SQL identifier: {identifier!r}")
    return f"`{identifier}`"


def quote_identifier_path(value: str, max_parts: int = 2) -> str:
    parts = value.split(".")
    if not 1 <= len(parts) <= max_parts or any(not part for part in parts):
        raise ValueError(f"Invalid SQL identifier path: {value!r}")
    return ".".join(_quote_identifier(part) for part in parts)


def validate_search_index_name(index_name: str) -> str:
    if not INDEX_NAME_RE.fullmatch(index_name):
        raise ValueError(f"Invalid search index name: {index_name!r}")
    if index_name in {"_all", ".", ".."} or "*" in index_name or "," in index_name:
        raise ValueError(f"Unsafe search index name: {index_name!r}")
    return index_name


def validate_search_field_name(field_name: str) -> str:
    if not FIELD_PATH_RE.fullmatch(field_name):
        raise ValueError(f"Invalid search field name: {field_name!r}")
    return field_name


def validate_lindorm_instance_id(instance_id: str) -> str:
    if not LINDORM_INSTANCE_ID_RE.fullmatch(instance_id):
        raise ValueError(f"Invalid Lindorm instance ID: {instance_id!r}")
    return instance_id


def validate_model_name(model_name: str) -> str:
    if not MODEL_NAME_RE.fullmatch(model_name):
        raise ValueError(f"Invalid model name: {model_name!r}")
    return model_name


def validate_api_key(api_key: str) -> str:
    if len(api_key) < MIN_API_KEY_LENGTH:
        raise ValueError(f"API_KEY must be at least {MIN_API_KEY_LENGTH} characters")
    return api_key


def validate_positive_limit(value: int, name: str, maximum: int = 100) -> int:
    if not isinstance(value, int) or value < 1 or value > maximum:
        raise ValueError(f"{name} must be an integer between 1 and {maximum}")
    return value


def validate_query_text(query_text: str) -> str:
    if not isinstance(query_text, str):
        raise ValueError("query_text must be a string")
    if not query_text:
        raise ValueError("query_text must not be empty")
    if len(query_text) > MAX_QUERY_TEXT_LENGTH:
        raise ValueError(f"query_text must be at most {MAX_QUERY_TEXT_LENGTH} characters")
    return query_text


def validate_readonly_select(sql: str) -> str:
    if "\x00" in sql:
        raise ValueError("SQL contains a null byte")

    statement = sql.strip()
    if not statement:
        raise ValueError("SQL query must not be empty")

    policy_sql = _strip_sql_comments_and_literals(statement).strip()
    if policy_sql.endswith(";"):
        policy_sql = policy_sql[:-1].strip()
        statement = statement.rstrip().rstrip(";").rstrip()

    if ";" in policy_sql:
        raise ValueError("Multiple SQL statements are not allowed")

    tokens = [token.lower() for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", policy_sql)]
    if not tokens or tokens[0] != "select":
        raise ValueError("Only SELECT queries are allowed")

    forbidden = sorted(set(tokens).intersection(FORBIDDEN_READONLY_TOKENS))
    if forbidden:
        raise ValueError(f"Forbidden token in read-only query: {forbidden[0]}")

    return statement


def _strip_sql_comments_and_literals(sql: str) -> str:
    chars: list[str] = []
    i = 0
    in_single = False
    in_double = False
    in_backtick = False
    in_line_comment = False
    in_block_comment = False

    while i < len(sql):
        current = sql[i]
        next_char = sql[i + 1] if i + 1 < len(sql) else ""

        if in_line_comment:
            if current in "\r\n":
                in_line_comment = False
                chars.append(current)
            else:
                chars.append(" ")
            i += 1
            continue

        if in_block_comment:
            if current == "*" and next_char == "/":
                chars.extend("  ")
                in_block_comment = False
                i += 2
            else:
                chars.append(" ")
                i += 1
            continue

        if in_single:
            if current == "\\" and next_char:
                chars.extend("  ")
                i += 2
                continue
            if current == "'":
                in_single = False
            chars.append(" ")
            i += 1
            continue

        if in_double:
            if current == "\\" and next_char:
                chars.extend("  ")
                i += 2
                continue
            if current == '"':
                in_double = False
            chars.append(" ")
            i += 1
            continue

        if in_backtick:
            if current == "`":
                in_backtick = False
            chars.append(" ")
            i += 1
            continue

        if current == "-" and next_char == "-":
            in_line_comment = True
            chars.extend("  ")
            i += 2
            continue
        if current == "#":
            in_line_comment = True
            chars.append(" ")
            i += 1
            continue
        if current == "/" and next_char == "*":
            in_block_comment = True
            chars.extend("  ")
            i += 2
            continue
        if current == "'":
            in_single = True
            chars.append(" ")
            i += 1
            continue
        if current == '"':
            in_double = True
            chars.append(" ")
            i += 1
            continue
        if current == "`":
            in_backtick = True
            chars.append(" ")
            i += 1
            continue

        chars.append(current)
        i += 1

    return "".join(chars)


def validate_required(values: Iterable[tuple[str, str | None]]) -> None:
    missing = [name for name, value in values if value is None or value == ""]
    if missing:
        raise ValueError(f"Missing required configuration: {', '.join(missing)}")
