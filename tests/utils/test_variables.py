from formulaic.utils.layered_mapping import LayeredMapping
from formulaic.utils.variables import Variable, get_expression_variables


def test_get_expression_variables():
    assert get_expression_variables("a + b", {}) == {"a", "b"}
    variables = get_expression_variables(
        "a + b * c.d",
        LayeredMapping(
            {"a": 1},
            LayeredMapping(
                {"b": 1},
                LayeredMapping({"c": 1}, name="subchild"),
                name="child",
            ),
        ),
    )
    assert variables == {"a", "b", "c.d"}
    vd = {variable: variable for variable in variables}
    assert vd["a"].source is None
    assert vd["b"].source == "child"
    assert vd["c.d"].source == "child:subchild"


class test_variable:
    v_a = Variable("a", roles=["value"], source="data")
    v_a_2 = Variable("a", roles=["callable"], source="data")
    v_b = Variable("b", roles=["value"], source="data")

    combined = {v: v for v in Variable.union({v_a}, {v_a_2, v_b})}
    assert combined["a"].roles == {"value", "callable"}
    assert combined["a"].source == "data"
    assert combined["b"].roles == {"value"}
    assert combined["b"].source == "data"
    assert len(combined) == 2
