from enum import Enum
import inspect
import json
import logging
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ValidationError as ValidationError
import pytest
from typing_extensions import TypedDict

from yaaal.core.tool import (
    Tool,
    anthropic_pydantic_function_tool,
    extract_function_description,
    extract_param_descriptions,
    function_schema,
    tool,
)
from yaaal.types_.core import FunctionSchema, ToolResultMessage


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

    def test_function_without_annotations(self):
        # Function without parameter annotations should fall back to Any.
        def no_annotation(a, b):
            """No parameter annotations; returns the first argument."""
            return a

        fs = function_schema(no_annotation)
        # Since 'a' is Any, any value is acceptable.
        args, kwargs = fs.to_call_args({"a": "hello", "b": [1, 2, 3]})
        result = no_annotation(*args, **kwargs)
        assert result == "hello"

    def test_function_with_mixed_annotations_invalid(self):
        # Mixed annotations: b is annotated as int while a is unannotated.
        def mixed(a, b: int):
            """Adds a and b, where b must be an integer."""
            return a + b

        fs = function_schema(mixed)
        # Valid input: b is an int.
        args, kwargs = fs.to_call_args({"a": 10, "b": 5})
        result = mixed(*args, **kwargs)
        assert result == 15

        # Invalid input: b is provided as a string, which cannot be coerced to int.
        with pytest.raises(ValidationError):
            fs.model_validate({"a": 10, "b": "non-int"})


class TestToolDecorator:
    @pytest.fixture
    def add3(self):
        @tool
        def add3(x: int, y: int) -> int:
            """Add two numbers then add 3."""
            return x + y + 3

        return add3

    @pytest.fixture
    def name3(self):
        @tool
        def name3(x: int, name: str = "default") -> str:
            """Concatenate name and number with 3."""
            return f"{name}{x}3"

        return name3

    def test_tool_decorator_call(self, add3, name3):
        assert add3(x=1, y=2) == 6  # 1 + 2 + 3
        assert name3(x=1) == "default13"
        assert name3(x=1, name="bob") == "bob13"

    def test_tool_decorator_schema(self, add3, name3):
        # Test add3 schema
        schema_add3 = add3.function_schema.json_schema
        assert schema_add3["title"] == "add3"
        assert "x" in schema_add3["properties"]
        assert schema_add3["properties"]["x"]["type"] == "integer"
        assert "y" in schema_add3["properties"]
        assert schema_add3["properties"]["y"]["type"] == "integer"
        assert set(schema_add3["required"]) == {"x", "y"}

        # Test name3 schema
        schema_name3 = name3.function_schema.json_schema
        assert schema_name3["title"] == "name3"
        assert "x" in schema_name3["properties"]
        assert schema_name3["properties"]["x"]["type"] == "integer"
        assert "name" in schema_name3["properties"]
        assert schema_name3["properties"]["name"]["type"] == "string"
        assert schema_name3["properties"]["name"]["default"] == "default"
        # assert set(schema_name3["required"]) == {"name", "x"}

    def test_tool_with_varargs(self):
        @tool
        def sum_numbers(*numbers: float, multiplier: float = 1.0) -> float:
            """Sum numbers and multiply by multiplier."""
            return sum(numbers) * multiplier

        assert sum_numbers(1.0, 2.0, 3.0) == 6.0
        assert sum_numbers(1.0, 2.0, multiplier=2.0) == 6.0

        schema = sum_numbers.function_schema.json_schema
        assert schema["properties"]["numbers"]["type"] == "array"
        assert schema["properties"]["numbers"]["items"]["type"] == "number"
        assert schema["properties"]["multiplier"]["type"] == "number"
        assert schema["properties"]["multiplier"]["default"] == 1.0

    def test_tool_with_kwargs(self):
        @tool
        def format_string(template: str, **kwargs: str) -> str:
            """Format string with kwargs."""
            return template.format(**kwargs)

        assert format_string("Hello {name}!", name="World") == "Hello World!"

        schema = format_string.function_schema.json_schema
        assert schema["properties"]["template"]["type"] == "string"
        assert schema["properties"]["kwargs"]["type"] == "object"
        assert schema["properties"]["kwargs"]["additionalProperties"]["type"] == "string"

    def test_tool_with_pydantic_models(self):
        class Inner(BaseModel):
            value: int

        class Outer(BaseModel):
            inner: Inner
            name: str

        @tool
        def nested_model(outer: Outer) -> dict:
            """Process nested model."""
            return {"name": outer.name, "value": outer.inner.value}

        result = nested_model(Outer.model_validate({"inner": {"value": 42}, "name": "test"}))
        assert result == {"name": "test", "value": 42}

        schema = nested_model.function_schema.json_schema
        assert "$defs" in schema
        assert "Inner" in schema["$defs"]
        assert "Outer" in schema["$defs"]

    def test_tool_return_validation(self):
        @tool(returns=int)
        def str_to_int(value: str):
            """Return string that should be converted to int by the tool decorator."""
            return value

        assert str_to_int(value="42") == 42  # Auto-converts string to int

        with pytest.raises(TypeError):
            str_to_int(value="not a number")

    def test_tool_with_union_return(self):
        @tool
        def maybe_int(value: str) -> int | str:
            """Return int if possible, otherwise string."""
            try:
                return int(value)
            except ValueError:
                return value

        assert maybe_int(value="42") == 42
        assert maybe_int(value="hello world") == "hello world"

    def test_tool_respond_as_tool(self):
        # Test string response
        result = Tool.respond_as_tool("test_id", "hello")
        assert isinstance(result, ToolResultMessage)
        assert result.tool_call_id == "test_id"
        assert result.content == "hello"

        # Test dict response
        result = Tool.respond_as_tool("test_id", {"key": "value"})
        assert json.loads(result.content) == {"key": "value"}

        # Test Pydantic model response
        class TestModel(BaseModel):
            value: str = "test"

        result = Tool.respond_as_tool("test_id", TestModel())
        assert json.loads(result.content) == {"value": "test"}

        # Test missing tool_call_id
        with pytest.raises(ValueError):
            Tool.respond_as_tool(None, "test")


# Unit tests for Tool return type validation and coercion
class TestToolReturnValidation:
    def test_tool_without_return_type(self):
        def no_return(x: int):
            pass

        t = Tool(no_return)
        assert t.returns is Any
        assert t(1) is None

    def test_tool_respond_as_tool_with_string(self):
        result = Tool.respond_as_tool("test_id", "test response")
        assert isinstance(result, ToolResultMessage)
        assert result.tool_call_id == "test_id"
        assert result.content == "test response"

    def test_tool_respond_as_tool_with_pydantic(self):
        class TestModel(BaseModel):
            value: str = "test"

        result = Tool.respond_as_tool("test_id", TestModel())
        assert json.loads(result.content) == {"value": "test"}

    def test_tool_respond_as_tool_with_dict(self):
        data = {"key": "value"}
        result = Tool.respond_as_tool("test_id", data)
        assert isinstance(result, ToolResultMessage)
        assert json.loads(result.content) == data

    def test_tool_respond_as_tool_with_non_serializable(self):
        class NonSerializable:
            def __str__(self):
                return "test object"

        obj = NonSerializable()
        result = Tool.respond_as_tool("test_id", obj)
        assert isinstance(result, ToolResultMessage)
        assert result.content == "test object"

    def test_tool_respond_as_tool_missing_id(self):
        with pytest.raises(ValueError, match="tool_call_id is required"):
            Tool.respond_as_tool(None, "test")

    def test_tool_wraps_function_metadata(self):
        def test_fn(x: int) -> str:
            """Test function"""
            return str(x)

        t = Tool(test_fn)
        assert t.__name__ == "test_fn"
        assert t.__doc__ == "Test function"

    def test_tool_return_union_coercion(self):
        def fn(x: str) -> int | str:
            return x

        t = Tool(fn)
        assert t(2) == 2
        assert t("abc") == "abc"

    def test_tool_return_mixed_coercion(self):
        def fn(x: int) -> int | str:
            if x % 2 == 0:
                return "123"  # coercible to int
            else:
                return 456

        t = Tool(fn)
        assert t(2) == 123
        assert t(3) == 456

    def test_tool_union_invalid_value(self):
        def fn(x: int) -> int:
            return [1, 2, 3]  # not coercible

        t = Tool(fn)
        with pytest.raises(TypeError):
            t(1)

    def test_validate_return_type_error(self):
        def fn(x: int) -> int:
            return "not an int"

        t = Tool(fn)
        with pytest.raises(TypeError):
            t(1)

    def test_tool_with_none_return_should_fail(self):
        def fn(x: int) -> int:
            return None

        t = Tool(fn)
        with pytest.raises(TypeError):
            t(1)

    def test_tool_with_invalid_model_return(self):
        class ExampleModel(BaseModel):
            value: int

        def fn(x: int) -> ExampleModel:
            return {"value": "not an int"}

        t = Tool(fn, returns=ExampleModel)
        with pytest.raises(TypeError):
            t(1)


# Integration tests for anthropic tool conversion
class TestAnthropicIntegration:
    @pytest.fixture
    def sample_model(self):
        class SampleModel(BaseModel):
            """This is a test."""

            x: int
            y: int

        return SampleModel

    def test_anthropic_pydantic_function_tool(self, sample_model):
        result = anthropic_pydantic_function_tool(sample_model)
        assert result["name"] == sample_model.__name__
        assert "description" in result
        props = result["input_schema"]["properties"]
        required = result["input_schema"]["required"]
        assert set(props.keys()) == {"x", "y"}
        assert set(required) == {"x", "y"}
