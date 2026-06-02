from formulaic.utils.code import sanitize_variable_names


def test_sanitize_variable_names():
    assert sanitize_variable_names("`1a`", {}, {}) == "_1a"
    assert sanitize_variable_names("`a b`", {}, {}) == "a_b"
    assert sanitize_variable_names("`z?!` + `z??`", {}, {}).startswith("z__  +  z___")
