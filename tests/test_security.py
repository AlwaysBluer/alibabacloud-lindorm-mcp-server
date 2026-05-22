import pytest

from src.lindorm_mcp_server.security import (
    quote_identifier_path,
    validate_api_key,
    validate_lindorm_instance_id,
    validate_model_name,
    validate_positive_limit,
    validate_query_text,
    validate_readonly_select,
    validate_search_field_name,
    validate_search_index_name,
)


def test_validate_readonly_select_allows_single_select():
    assert validate_readonly_select("SELECT * FROM users WHERE note = ';' ;") == (
        "SELECT * FROM users WHERE note = ';'"
    )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1; DROP TABLE users",
        "/* comment */ DELETE FROM users",
        "SELECT * FROM users WHERE id IN (DELETE FROM audit)",
        "",
        "SHOW TABLES",
    ],
)
def test_validate_readonly_select_rejects_unsafe_sql(sql):
    with pytest.raises(ValueError):
        validate_readonly_select(sql)


def test_quote_identifier_path_quotes_safe_table_names():
    assert quote_identifier_path("schema.table") == "`schema`.`table`"


@pytest.mark.parametrize("name", ["users;drop", "schema.table.extra", "`users`", "bad-name"])
def test_quote_identifier_path_rejects_unsafe_table_names(name):
    with pytest.raises(ValueError):
        quote_identifier_path(name)


def test_validate_search_inputs_allow_safe_values():
    assert validate_search_index_name("docs_index-1") == "docs_index-1"
    assert validate_search_field_name("metadata.content") == "metadata.content"
    assert validate_positive_limit(10, "top_k") == 10
    assert validate_lindorm_instance_id("ld-12345678") == "ld-12345678"
    assert validate_model_name("bge_model-1") == "bge_model-1"
    assert validate_api_key("a" * 32) == "a" * 32
    assert validate_query_text("hello") == "hello"


@pytest.mark.parametrize("index_name", ["*", "_all", "docs,index", "Docs"])
def test_validate_search_index_name_rejects_unsafe_values(index_name):
    with pytest.raises(ValueError):
        validate_search_index_name(index_name)


@pytest.mark.parametrize("field_name", ["content.keyword", "content;drop", "content[*]"])
def test_validate_search_field_name_rejects_unsafe_values(field_name):
    if field_name == "content.keyword":
        assert validate_search_field_name(field_name) == field_name
    else:
        with pytest.raises(ValueError):
            validate_search_field_name(field_name)


@pytest.mark.parametrize("limit", [0, 101, "10"])
def test_validate_positive_limit_rejects_bad_limits(limit):
    with pytest.raises(ValueError):
        validate_positive_limit(limit, "top_k")


@pytest.mark.parametrize("instance_id", ["127.0.0.1#", "ld-", "abc-12345678", "ld-../../x"])
def test_validate_lindorm_instance_id_rejects_unsafe_values(instance_id):
    with pytest.raises(ValueError):
        validate_lindorm_instance_id(instance_id)


@pytest.mark.parametrize("model_name", ["../model", "model/name", "", "-bad"])
def test_validate_model_name_rejects_unsafe_values(model_name):
    with pytest.raises(ValueError):
        validate_model_name(model_name)


def test_validate_api_key_rejects_short_values():
    with pytest.raises(ValueError):
        validate_api_key("short")


@pytest.mark.parametrize("query_text", ["", "x" * 4097, None])
def test_validate_query_text_rejects_bad_values(query_text):
    with pytest.raises(ValueError):
        validate_query_text(query_text)
