from enum import Enum
import inspect
import json
import logging
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ValidationError
import pytest
from typing_extensions import TypedDict

from yaaal.core.tool import (
    FunctionSchema,
    Tool,
    anthropic_pydantic_function_tool,
    extract_function_description,
    extract_param_descriptions,
    function_schema,
    tool,
)
from yaaal.types_.core import ToolResultMessage


class TestExtractFunctionDescription:
    def test_no_docstring(self):
        def no_doc(a: int):
            return a

        # No docstring returns None
        assert extract_function_description(no_doc) is None

    def test_google_style_docstring(self):
        def fn(a: int) -> int:
            """
            This function adds a number.

            Args:
                a: number to be added.
            """
            return a

        # Expect the description to be extracted as the leading text.
        desc = extract_function_description(fn)
        # Strip to avoid leading/trailing whitespace issues.
        assert desc is not None
        assert desc.strip() == "This function adds a number."

    def test_sphinx_style_docstring(self):
        def fn(a: int) -> int:
            """
            This function multiplies a number.

            :param a: integer to multiply.
            """
            return a

        desc = extract_function_description(fn)
        assert desc is not None
        assert desc.strip() == "This function multiplies a number."

    def test_numpy_style_docstring(self):
        def fn(a: int) -> int:
            """
            Multiply a number by two.

            Parameters
            ----------
            a : int
                The number to be multiplied.
            """
            return a

        desc = extract_function_description(fn)
        assert desc is not None
        assert desc.strip() == "Multiply a number by two."


class TestExtractParamDescriptions:
    def test_no_docstring(self):
        def no_doc(a: int):
            return a

        # With no docstring, expect an empty dict.
        params = extract_param_descriptions(no_doc)
        assert params == {}

    def test_google_style_parameters(self):
        def fn(a: int, b: str) -> int:
            """
            Concatenate parameter descriptions.

            Args:
                a: description for parameter a.
                b: description for parameter b.
            """
            return a

        params = extract_param_descriptions(fn)
        assert "a" in params
        assert "b" in params
        assert params["a"].strip() == "description for parameter a."
        assert params["b"].strip() == "description for parameter b."

    def test_sphinx_style_parameters(self):
        def fn(a: int) -> int:
            """
            Sphinx style parameter description.

            :param a: parameter a in sphinx style.
            """
            return a

        params = extract_param_descriptions(fn)
        assert "a" in params
        assert "parameter a in sphinx style" in params["a"]

    def test_numpy_style_parameters(self):
        def fn(a: int) -> int:
            """
            Numpy style parameter description.

            Parameters
            ----------
            a : int
                The parameter a to be processed.
            """
            return a

        params = extract_param_descriptions(fn)
        assert "a" in params
        # Check that extracted description contains expected text.
        assert "parameter a" in params["a"] or "to be processed" in params["a"]


# Unit tests for function signature extraction and coercion
class TestFunctionSchema:
    """Test suite for function_schema function."""

    def test_no_args_function(self):
        def no_args_function():
            """This function has no args."""
            return "ok"

        fs = function_schema(no_args_function)

        assert isinstance(fs, FunctionSchema)
        assert isinstance(fs.json_schema, dict)
        assert fs.json_schema.get("title") == "no_args_function"
        assert fs.json_schema.get("description") == "This function has no args."

        args, kwargs_dict = fs.to_call_args({})
        result = no_args_function(*args, **kwargs_dict)
        assert result == "ok"

    def test_simple_function(self):
        """Test a function that has simple typed parameters and defaults."""

        def simple_function(a: int, b: int = 5):
            """
            Args:
                a: The first argument
                b: The second argument

            Returns:
                The sum of a and b
            """
            return a + b

        fs = function_schema(simple_function)
        assert isinstance(fs, FunctionSchema)

        assert isinstance(fs.json_schema, dict)
        assert fs.json_schema.get("title") == "simple_function"
        assert fs.json_schema.get("properties", {}).get("a").get("description") == "The first argument"
        assert fs.json_schema.get("properties", {}).get("b").get("description") == "The second argument"

        # Valid input
        valid_input = {"a": 3}
        args_tuple, kwargs_dict = fs.to_call_args(valid_input)
        result = simple_function(*args_tuple, **kwargs_dict)
        assert result == 8  # 3 + 5

        # Another valid input
        valid_input2 = {"a": 3, "b": 10}
        args_tuple2, kwargs_dict2 = fs.to_call_args(valid_input2)
        result2 = simple_function(*args_tuple2, **kwargs_dict2)
        assert result2 == 13  # 3 + 10

        # Invalid input: 'a' must be int
        with pytest.raises(ValidationError):
            fs.model_validate({"a": "not an integer"})

    def test_varargs_function(self):
        """Test a function that uses *args and **kwargs."""

        def varargs_function(x: int, *numbers: float, flag: bool = False, **kwargs: Any):
            """A function with varargs and kwargs."""
            return x, numbers, flag, kwargs

        fs = function_schema(varargs_function)

        # Check JSON schema structure
        assert isinstance(fs.json_schema, dict)
        assert fs.json_schema.get("title") == "varargs_function"

        # Valid input including *args in 'numbers' and **kwargs in 'kwargs'
        valid_input = {
            "x": 10,
            "numbers": [1.1, 2.2, 3.3],
            "flag": True,
            "kwargs": {"extra1": "hello", "extra2": 42},
        }
        args, kwargs = fs.to_call_args(valid_input)
        result = varargs_function(*args, **kwargs)
        assert result[0] == 10
        assert result[1] == (1.1, 2.2, 3.3)
        assert result[2] is True
        assert result[3] == {"extra1": "hello", "extra2": 42}

        # Missing 'x' should raise error
        with pytest.raises(ValidationError):
            fs.model_validate({"numbers": [1.1, 2.2]})

        # 'flag' can be omitted because it has a default
        valid_input_no_flag = {"x": 7, "numbers": [9.9], "kwargs": {"some_key": "some_value"}}
        args2, kwargs2 = fs.to_call_args(valid_input_no_flag)
        result2 = varargs_function(*args2, **kwargs2)
        assert result2 == (7, (9.9,), False, {"some_key": "some_value"})

    def test_nested_data_function(self):
        class Foo(TypedDict):
            a: int
            b: str

        class InnerModel(BaseModel):
            a: int
            b: str

        class OuterModel(BaseModel):
            inner: InnerModel
            foo: Foo

        def complex_args_function(model: OuterModel) -> str:
            return f"{model.inner.a}, {model.inner.b}, {model.foo['a']}, {model.foo['b']}"

        fs = function_schema(complex_args_function)
        assert isinstance(fs.json_schema, dict)
        assert fs.json_schema.get("title") == "complex_args_function"

        # Valid input
        model = OuterModel(inner=InnerModel(a=1, b="hello"), foo={"a": 2, "b": "world"})
        valid_input = {"model": model.model_dump()}

        args, kwargs = fs.to_call_args(valid_input)
        result = complex_args_function(*args, **kwargs)
        assert result == "1, hello, 2, world"

    def test_complex_args_and_docs_function(self):
        """Test a function with complex args and detailed docstrings."""

        class Foo(TypedDict):
            a: int
            b: str

        class InnerModel(BaseModel):
            a: int
            b: str

        class OuterModel(BaseModel):
            inner: InnerModel
            foo: Foo

        def complex_args_and_docs_function(model: OuterModel, some_flag: int = 0) -> str:
            """
            This function takes a model and a flag, and returns a string.

            Args:
                model: A model with an inner and foo field
                some_flag: An optional flag with a default of 0

            Returns:
                A string with the values of the model and flag
            """
            return f"{model.inner.a}, {model.inner.b}, {model.foo['a']}, {model.foo['b']}, {some_flag or 0}"

        fs = function_schema(complex_args_and_docs_function)

        # Check JSON schema structure
        assert isinstance(fs.json_schema, dict)
        assert fs.json_schema.get("title") == "complex_args_and_docs_function"

        # Check docstring parsing
        properties = fs.json_schema.get("properties", {})
        assert properties.get("model").get("description") == "A model with an inner and foo field"
        assert properties.get("some_flag").get("description") == "An optional flag with a default of 0"

        # Valid input
        model = OuterModel(inner=InnerModel(a=1, b="hello"), foo={"a": 2, "b": "world"})
        valid_input = {"model": model.model_dump()}

        args, kwargs = fs.to_call_args(valid_input)
        result = complex_args_and_docs_function(*args, **kwargs)
        assert result == "1, hello, 2, world, 0"

        # Invalid input: 'some_flag' must be int
        with pytest.raises(ValidationError):
            fs.model_validate({"model": model.model_dump(), "some_flag": "not an int"})

    def test_class_based_functions(self):
        """Test handling of instance, class, and static methods."""

        class MyClass:
            def method(self, a: int, b: int = 5) -> int:
                """Instance method that adds numbers."""
                return a + b

            @classmethod
            def cls_method(cls, a: int, b: int = 5) -> int:
                """Class method that adds numbers."""
                return a + b

            @staticmethod
            def static_method(a: int, b: int = 5) -> int:
                """Static method that adds numbers."""
                return a + b

        instance = MyClass()

        # Instance method
        fs = function_schema(instance.method)
        assert isinstance(fs.json_schema, dict)
        assert fs.json_schema.get("title") == "method"

        valid_input = {"a": 1, "b": 2}
        fs.model_validate(valid_input)
        args, kwargs = fs.to_call_args(valid_input)
        result = instance.method(*args, **kwargs)
        assert result == 3

        # Class method
        fs2 = function_schema(MyClass.cls_method)
        assert isinstance(fs2.json_schema, dict)
        assert fs2.json_schema.get("title") == "cls_method"

        args, kwargs = fs2.to_call_args(valid_input)
        class_result = MyClass.cls_method(*args, **kwargs)
        instance_result = instance.cls_method(*args, **kwargs)
        assert class_result == instance_result == 3

        # Static method
        fs3 = function_schema(MyClass.static_method)
        assert isinstance(fs3.json_schema, dict)
        assert fs3.json_schema.get("title") == "static_method"

        args, kwargs = fs3.to_call_args(valid_input)
        class_result = MyClass.static_method(*args, **kwargs)
        instance_result = instance.static_method(*args, **kwargs)
        assert class_result == instance_result == 3

    def test_enum_and_literal_function(self):
        class MyEnum(str, Enum):
            FOO = "foo"
            BAR = "bar"
            BAZ = "baz"

        def enum_and_literal_function(a: MyEnum, b: Literal["a", "b", "c"]) -> str:
            return f"{a.value} {b}"

        fs = function_schema(enum_and_literal_function)
        assert isinstance(fs.json_schema, dict)
        assert fs.json_schema.get("title") == "enum_and_literal_function"

        # Check that the enum values are included in the JSON schema
        assert fs.json_schema.get("$defs", {}).get("MyEnum", {}).get("enum") == [
            "foo",
            "bar",
            "baz",
        ]

        # Check that the enum is expressed as a def
        assert fs.json_schema.get("properties", {}).get("a", {}).get("$ref") == "#/$defs/MyEnum"

        # Check that the literal values are included in the JSON schema
        assert fs.json_schema.get("properties", {}).get("b", {}).get("enum") == [
            "a",
            "b",
            "c",
        ]

        # Valid input
        valid_input = {"a": "foo", "b": "a"}
        args, kwargs = fs.to_call_args(valid_input)
        result = enum_and_literal_function(*args, **kwargs)
        assert result == "foo a"

        # Invalid input: 'a' must be a valid enum value
        with pytest.raises(ValidationError):
            fs.model_validate({"a": "not an enum value", "b": "a"})

        # Invalid input: 'b' must be a valid literal value
        with pytest.raises(ValidationError):
            fs.model_validate({"a": "foo", "b": "not a literal value"})

    def test_var_positional_tuple_annotation(self):
        # When a function has a var-positional parameter annotated with a tuple type,
        # function_schema
        # () should convert it into a field with type List[<tuple-element>].
        def func(*args: tuple[int, ...]) -> int:
            total = 0
            for arg in args:
                total += sum(arg)
            return total

        fs = function_schema(func)

        properties = fs.json_schema.get("properties", {})
        assert properties.get("args").get("type") == "array"
        assert properties.get("args").get("items").get("type") == "integer"

    def test_var_keyword_dict_annotation(self):
        # Case 3:
        # When a function has a var-keyword parameter annotated with a dict type,
        # function_schema
        # () should convert it into a field with type Dict[<key>, <value>].
        def func(**kwargs: dict[str, int]):
            return kwargs

        fs = function_schema(func)

        properties = fs.json_schema.get("properties", {})
        # The name of the field is "kwargs", and it's a JSON object i.e. a dict.
        assert properties.get("kwargs").get("type") == "object"
        # The values in the dict are integers.
        assert properties.get("kwargs").get("additionalProperties").get("type") == "integer"


#     def test_basic_function(self):
#         def sample_fn(x: int, y: str = "default") -> None:
#             """Sample function with basic types."""
#             pass

#         model = function_schema
# (sample_fn)
#         schema = model.model_json_schema()
#         assert schema["title"] == "sample_fn"
#         assert schema["description"] == "Sample function with basic types."
#         assert "x" in schema["properties"]
#         assert schema["properties"]["x"]["type"] == "integer"
#         assert "y" in schema["properties"]
#         assert schema["properties"]["y"]["type"] == "string"
#         assert schema["properties"]["y"]["default"] == "default"
#         assert "x" in schema["required"]
#         assert "y" not in schema["required"]

#     def test_function_with_union(self):
#         def union_fn(x: Union[int, str]) -> None:
#             pass

#         model = function_schema
# (union_fn)
#         assert issubclass(model, BaseModel)

#         schema = model.model_json_schema()
#         assert "anyOf" in schema["properties"]["x"]
#         assert len(schema["properties"]["x"]["anyOf"]) == 2

#     def test_function_with_uniontype(self):
#         def union_fn(x: int | str) -> None:
#             pass

#         model = function_schema
# (union_fn)
#         assert issubclass(model, BaseModel)

#         schema = model.model_json_schema()
#         assert "anyOf" in schema["properties"]["x"]
#         assert len(schema["properties"]["x"]["anyOf"]) == 2

#     def test_function_with_optional(self):
#         def optional_fn(x: Optional[int] = None, y: str = "default") -> None:
#             pass

#         model = function_schema
# (optional_fn)
#         assert issubclass(model, BaseModel)

#         schema = model.model_json_schema()
#         assert "x" in schema["properties"]
#         assert "anyOf" in schema["properties"]["x"]
#         assert len(schema["properties"]["x"]["anyOf"]) == 2
#         assert "y" in schema["properties"]
#         assert schema["properties"]["y"]["default"] == "default"

#     def test_args_kwargs(self):
#         def variadic_fn(*args: int, **kwargs: str) -> None:
#             pass

#         model = function_schema
# (variadic_fn)
#         schema = model.model_json_schema()
#         assert "args_list" in schema["properties"]
#         assert "kwargs_dict" in schema["properties"]

#     def test_class_method(self):
#         class SampleClass:
#             def method(self, x: int) -> None:
#                 pass

#         instance = SampleClass()
#         model = function_schema
# (instance.method)
#         assert issubclass(model, BaseModel)

#         schema = model.model_json_schema()
#         assert schema["title"] == "method"
#         assert "x" in schema["properties"]
#         assert "self" not in schema["properties"]

#     def test_function_with_docstring(self):
#         def documented_fn(x: int) -> None:
#             """Test documentation."""
#             pass

#         model = function_schema
# (documented_fn)
#         schema = model.model_json_schema()
#         assert schema["description"] == "Test documentation."

#     def test_function_without_types(self, caplog):
#         def untyped_fn(x, y):
#             pass

#         caplog.set_level(logging.INFO, logger="yaaal.core.tool")

#         _ = function_schema
# (untyped_fn)
#         logs = caplog.record_tuples

#         assert any("No type annotation provided" in lg[2] for lg in logs)

#     def test_function_with_missing_docstring(self, caplog):
#         def undocumented_fn(x: int) -> None:
#             pass

#         caplog.set_level(logging.INFO, logger="yaaal.core.tool")

#         model = function_schema
# (undocumented_fn)
#         schema = model.model_json_schema()
#         assert "description" not in schema

#         logs = caplog.record_tuples
#         assert any("requires docstrings for viable signature" in lg[2] for lg in logs)

#     def test_function_with_complex_types(self):
#         class NestedModel(BaseModel):
#             value: int

#         def complex_fn(x: list[NestedModel], y: dict[str, int]) -> None:
#             pass

#         model = function_schema
# (complex_fn)
#         schema = model.model_json_schema()
#         assert "NestedModel" in schema.get("$defs", {})
#         assert schema["properties"]["x"]["type"] == "array"
#         assert schema["properties"]["y"]["type"] == "object"

#     def test_class_method_with_cls(self):
#         class TestClass:
#             @classmethod
#             def class_method(cls, x: int) -> None:
#                 pass

#         model = function_schema
# (TestClass.class_method)
#         schema = model.model_json_schema()
#         assert "x" in schema["properties"]
#         assert "cls" not in schema["properties"]

#     def test_function_with_enum(self):
#         class Color(str, Enum):
#             RED = "red"
#             BLUE = "blue"

#         def color_fn(color: Color) -> None:
#             pass

#         model = function_schema
# (color_fn)
#         schema = model.model_json_schema()
#         assert "$defs" in schema
#         assert "Color" in schema["$defs"]
#         assert schema["$defs"]["Color"]["enum"] == ["red", "blue"]

#     def test_function_with_literal(self):
#         def direction_fn(dir: Literal["north", "south"]) -> None:
#             pass

#         model = function_schema
# (direction_fn)
#         schema = model.model_json_schema()
#         assert schema["properties"]["dir"]["enum"] == ["north", "south"]

#     def test_nested_models(self):
#         class Inner(BaseModel):
#             value: int

#         class Outer(BaseModel):
#             inner: Inner
#             name: str

#         def nested_fn(model: Outer) -> None:
#             pass

#         model = function_schema
# (nested_fn)
#         schema = model.model_json_schema()
#         assert "$defs" in schema
#         assert "Inner" in schema["$defs"]
#         assert "Outer" in schema["$defs"]

#     def test_varargs_kwargs(self):
#         def variadic_fn(*args: int, **kwargs: str) -> None:
#             pass

#         model = function_schema
# (variadic_fn)
#         schema = model.model_json_schema()
#         assert "args_list" in schema["properties"]
#         assert schema["properties"]["args_list"]["type"] == "array"
#         assert schema["properties"]["args_list"]["items"]["type"] == "integer"
#         assert "kwargs_dict" in schema["properties"]
#         assert schema["properties"]["kwargs_dict"]["type"] == "object"
#         assert schema["properties"]["kwargs_dict"]["additionalProperties"]["type"] == "string"

#     def test_function_with_tuple_args(self):
#         def tuple_fn(*args: tuple[int, ...]) -> None:
#             pass

#         model = function_schema
# (tuple_fn)
#         schema = model.model_json_schema()
#         assert schema["properties"]["args_list"]["type"] == "array"
#         assert schema["properties"]["args_list"]["items"]["type"] == "integer"

#     def test_method_types(self):
#         class Sample:
#             def instance_method(self, x: int) -> None:
#                 pass

#             @classmethod
#             def class_method(cls, x: int) -> None:
#                 pass

#             @staticmethod
#             def static_method(x: int) -> None:
#                 pass

#         instance = Sample()
#         for method in [instance.instance_method, Sample.class_method, Sample.static_method]:
#             model = function_schema
# (method)
#             schema = model.model_json_schema()
#             assert "x" in schema["properties"]
#             assert "self" not in schema["properties"]
#             assert "cls" not in schema["properties"]

#     def test_optional_and_union_types(self):
#         from typing import Optional, Union

#         def union_fn(x: Optional[int], y: Union[str, int]) -> None:
#             pass

#         model = function_schema
# (union_fn)
#         schema = model.model_json_schema()
#         assert "anyOf" in schema["properties"]["x"]
#         assert len(schema["properties"]["x"]["anyOf"]) == 2
#         assert "anyOf" in schema["properties"]["y"]
#         assert len(schema["properties"]["y"]["anyOf"]) == 2


# # Tests for tool decorator functionality
# class TestToolDecorator:
#     @pytest.fixture
#     def add3(self):
#         @tool
#         def add3(x: int, y: int) -> int:
#             return x + y + 3

#         return add3

#     @pytest.fixture
#     def name3(self):
#         @tool
#         def name3(x: int, name: str) -> str:
#             return f"{name}{x}3"

#         return name3

#     def test_tool_decorator_call(self, add3, name3):
#         assert add3(1, 2) == 6  # 1 + 2 + 3
#         assert name3(1, "bob") == "bob13"

#     def test_tool_decorator_signature(self, add3, name3):
#         model_add3 = add3.signature
#         schema_add3 = model_add3.model_json_schema()
#         assert schema_add3["title"] == "add3"
#         assert "x" in schema_add3["properties"]
#         assert schema_add3["properties"]["x"]["type"] == "integer"

#         model_name3 = name3.signature
#         schema_name3 = model_name3.model_json_schema()
#         assert schema_name3["title"] == "name3"
#         assert "name" in schema_name3["properties"]
#         assert schema_name3["properties"]["name"]["type"] == "string"


# # Unit tests for Tool return type validation and coercion
# class TestToolReturnValidation:
#     def test_tool_without_return_type(self):
#         def no_return(x: int):
#             pass

#         t = Tool(no_return)
#         assert t.returns is None

#     def test_tool_respond_as_tool_with_string(self):
#         result = Tool.respond_as_tool("test_id", "test response")
#         assert isinstance(result, ToolResultMessage)
#         assert result.tool_call_id == "test_id"
#         assert result.content == "test response"

#     def test_tool_respond_as_tool_with_pydantic(self):
#         class TestModel(BaseModel):
#             value: str = "test"

#         result = Tool.respond_as_tool("test_id", TestModel())
#         assert json.loads(result.content) == {"value": "test"}

#     def test_tool_respond_as_tool_with_dict(self):
#         data = {"key": "value"}
#         result = Tool.respond_as_tool("test_id", data)
#         assert isinstance(result, ToolResultMessage)
#         assert json.loads(result.content) == data

#     def test_tool_respond_as_tool_with_non_serializable(self):
#         class NonSerializable:
#             def __str__(self):
#                 return "test object"

#         obj = NonSerializable()
#         result = Tool.respond_as_tool("test_id", obj)
#         assert isinstance(result, ToolResultMessage)
#         assert result.content == "test object"

#     def test_tool_respond_as_tool_missing_id(self):
#         with pytest.raises(ValueError, match="tool_call_id is required"):
#             Tool.respond_as_tool(None, "test")

#     def test_tool_wraps_function_metadata(self):
#         def test_fn(x: int) -> str:
#             """Test function"""
#             return str(x)

#         t = Tool(test_fn)
#         assert t.__name__ == "test_fn"
#         assert t.__doc__ == "Test function"

#     def test_tool_return_union_coercion(self):
#         def fn(x: int) -> int | str:
#             if x % 2 == 0:
#                 return x
#             else:
#                 return f"{x}"

#         t = Tool(fn)
#         assert t(2) == 2
#         assert t(3) == "3"

#     def test_tool_return_mixed_coercion(self):
#         def fn(x: int) -> int | str:
#             if x % 2 == 0:
#                 return "123"  # coercible to int
#             else:
#                 return 456

#         t = Tool(fn)
#         assert t(2) == 123
#         assert t(3) == 456

#     def test_tool_union_invalid_value(self):
#         def fn(x: int) -> int | str:
#             return [1, 2, 3]  # not coercible

#         t = Tool(fn)
#         with pytest.raises(Exception):
#             t(1)

#     def test_validate_return_type_error(self):
#         def fn(x: int) -> int:
#             return "not an int"

#         t = Tool(fn)
#         with pytest.raises(ValueError):
#             t(1)

#     def test_tool_with_none_return_should_fail(self):
#         def fn(x: int) -> int:
#             return None

#         t = Tool(fn)
#         with pytest.raises(TypeError):
#             t(1)

#     def test_tool_with_invalid_model_return(self):
#         from pydantic import ValidationError as PydanticValidationError

#         class ExampleModel(BaseModel):
#             value: int

#         def fn(x: int) -> ExampleModel:
#             return {"value": "not an int"}

#         t = Tool(fn, returns=ExampleModel)
#         with pytest.raises(PydanticValidationError):
#             t(1)


# # Integration tests for anthropic tool conversion
# class TestAnthropicIntegration:
#     def test_anthropic_pydantic_function_tool(self, sample_model):
#         result = anthropic_pydantic_function_tool(sample_model)
#         assert result["name"] == sample_model.__name__
#         assert "description" in result
#         props = result["input_schema"]["properties"]
#         required = result["input_schema"]["required"]
#         assert set(props.keys()) == {"x", "y"}
#         assert set(required) == {"x", "y"}


# @pytest.fixture
# def context():
#     """Fixture for testing context parameters."""
#     from yaaal.types_.core import RunContextWrapper

#     return RunContextWrapper(context="test")


# @pytest.fixture
# def sample_enum():
#     """Fixture for testing enum parameters."""

#     class Color(str, Enum):
#         RED = "red"
#         BLUE = "blue"

#     return Color


# @pytest.fixture
# def nested_models():
#     """Fixture for testing nested model parameters."""

#     class Inner(BaseModel):
#         value: int

#     class Outer(BaseModel):
#         inner: Inner
#         name: str

#     return Inner, Outer


# class TestPydanticFunctionSignature:
#     """Test suite for function_schema
#  function.

#     Tests parameter handling, type validation, and schema generation for various
#     function signatures and parameter types.
#     """

#     def test_simple_function_parameters(self):
#         """Should correctly handle basic function parameters with types and defaults."""

#         def sample_fn(x: int, y: str = "default") -> None:
#             """Sample function docstring."""
#             pass

#         model = function_schema
# (sample_fn)
#         schema = model.model_json_schema()

#         assert schema["title"] == "sample_fn"
#         assert schema["description"] == "Sample function docstring."
#         assert schema["properties"]["x"]["type"] == "integer"
#         assert schema["properties"]["y"]["type"] == "string"
#         assert schema["properties"]["y"]["default"] == "default"
#         assert set(schema["required"]) == {"x"}

#     def test_no_type_annotations_logs_warning(self, caplog):
#         """Should log warning when function lacks type annotations."""

#         def untyped(x, y):
#             pass

#         caplog.set_level(logging.INFO)
#         _ = function_schema
# (untyped)
#         assert any("No type annotation provided" in record.message for record in caplog.records)

#     def test_varargs_kwargs_handling(self):
#         """Should properly handle *args and **kwargs patterns."""

#         def variadic(*numbers: float, **settings: bool) -> None:
#             pass

#         model = function_schema
# (variadic)
#         schema = model.model_json_schema()

#         assert schema["properties"]["numbers"]["type"] == "array"
#         assert schema["properties"]["numbers"]["items"]["type"] == "number"
#         assert schema["properties"]["settings"]["type"] == "object"
#         assert schema["properties"]["settings"]["additionalProperties"]["type"] == "boolean"

#     def test_tuple_varargs_conversion(self):
#         """Should convert tuple varargs to appropriate array type."""

#         def tuple_args(*args: tuple[int, ...]) -> None:
#             pass

#         model = function_schema
# (tuple_args)
#         schema = model.model_json_schema()

#         assert schema["properties"]["args_list"]["type"] == "array"
#         assert schema["properties"]["args_list"]["items"]["type"] == "integer"

#     def test_bound_method_handling(self):
#         """Should properly handle instance, class, and static methods."""

#         class Sample:
#             def method(self, x: int) -> None:
#                 pass

#             @classmethod
#             def cls_method(cls, y: str) -> None:
#                 pass

#             @staticmethod
#             def static_method(z: float) -> None:
#                 pass

#         instance = Sample()
#         methods = [instance.method, Sample.cls_method, Sample.static_method]

#         for method in methods:
#             model = function_schema
# (method)
#             schema = model.model_json_schema()
#             assert "self" not in schema["properties"]
#             assert "cls" not in schema["properties"]

#     def test_enum_parameter(self, sample_enum):
#         """Should properly handle Enum parameters."""

#         def color_fn(color: sample_enum) -> None:
#             pass

#         model = function_schema
# (color_fn)
#         schema = model.model_json_schema()

#         assert "$defs" in schema
#         assert schema["$defs"][sample_enum.__name__]["enum"] == ["red", "blue"]
#         assert schema["properties"]["color"]["$ref"] == f"#/$defs/{sample_enum.__name__}"

#     def test_literal_parameter(self):
#         """Should properly handle Literal type parameters."""

#         def direction_fn(dir: Literal["north", "south"]) -> None:
#             pass

#         model = function_schema
# (direction_fn)
#         schema = model.model_json_schema()

#         assert schema["properties"]["dir"]["enum"] == ["north", "south"]

#     def test_nested_model_parameters(self, nested_models):
#         """Should properly handle nested Pydantic model parameters."""
#         Inner, Outer = nested_models

#         def nested_fn(data: Outer) -> None:
#             pass

#         model = function_schema
# (nested_fn)
#         schema = model.model_json_schema()

#         assert "$defs" in schema
#         assert set(schema["$defs"].keys()) == {"Inner", "Outer"}
#         assert schema["properties"]["data"]["$ref"] == "#/$defs/Outer"

#     def test_union_types(self):
#         """Should properly handle Union and Optional types."""
#         from typing import Optional, Union

#         def union_fn(x: Union[int, str], y: Optional[float] = None, z: int | str = "default") -> None:
#             pass

#         model = function_schema
# (union_fn)
#         schema = model.model_json_schema()

#         for param in ["x", "z"]:
#             assert "anyOf" in schema["properties"][param]
#             assert len(schema["properties"][param]["anyOf"]) == 2

#         assert "anyOf" in schema["properties"]["y"]
#         assert None in [t.get("type") for t in schema["properties"]["y"]["anyOf"]]

#     def test_missing_docstring_logs_warning(self, caplog):
#         """Should log warning when function lacks docstring."""

#         def no_doc(x: int) -> None:
#             pass

#         caplog.set_level(logging.INFO)
#         _ = function_schema
# (no_doc)
#         assert any("requires docstrings" in record.message for record in caplog.records)
