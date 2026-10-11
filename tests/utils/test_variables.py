from formulaic.utils.layered_mapping import LayeredMapping
from formulaic.utils.stateful_transforms import stateful_transform
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


def test_get_expression_variables_stateful_transform_without_data():
    @stateful_transform
    def transform(data):
        raise AssertionError("Variable extraction must not evaluate the transform.")

    variables = get_expression_variables("transform(x + y)", {"transform": transform})
    assert variables == {"transform", "x", "y"}
    by_name = {str(variable): variable for variable in variables}
    assert by_name["transform"].roles == {Variable.Role.CALLABLE}
    assert by_name["x"].roles == {Variable.Role.VALUE}
    assert by_name["y"].roles == {Variable.Role.VALUE}


def test_get_expression_variables_stateful_transform_custom_getter():
    @stateful_transform(
        get_required_variables=lambda name: (Variable(name, roles=["value"]),)
    )
    def transform(data):
        raise AssertionError("Variable extraction must not evaluate the transform.")

    variables = get_expression_variables("transform('x?!')", {"transform": transform})
    assert variables == {"transform", "x?!"}
    by_name = {str(variable): variable for variable in variables}
    assert by_name["transform"].roles == {Variable.Role.CALLABLE}
    assert by_name["x?!"].roles == {Variable.Role.VALUE}


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
