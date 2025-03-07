import json
import logging
from typing import Optional, Union

from pydantic import BaseModel
import pytest

from yaaal.core.tool import Tool, anthropic_pydantic_function_tool, pydantic_function_signature, tool
from yaaal.types_.core import ToolResultMessage


class TestFunctionSchema:
    def test_basic_function(self):
        def sample_fn(x: int, y: str) -> None:
            pass

        model = pydantic_function_signature(sample_fn)
        assert issubclass(model, BaseModel)

        schema = model.model_json_schema()
        assert schema["title"] == "sample_fn"
        assert "x" in schema["properties"]
        assert schema["properties"]["x"]["type"] == "integer"
        assert "y" in schema["properties"]
        assert schema["properties"]["y"]["type"] == "string"
        assert set(schema["required"]) == {"x", "y"}

    def test_function_with_union(self):
        def union_fn(x: Union[int, str]) -> None:
            pass

        model = pydantic_function_signature(union_fn)
        assert issubclass(model, BaseModel)

        schema = model.model_json_schema()
        assert "anyOf" in schema["properties"]["x"]
        assert len(schema["properties"]["x"]["anyOf"]) == 2

    def test_function_with_uniontype(self):
        def union_fn(x: int | str) -> None:
            pass

        model = pydantic_function_signature(union_fn)
        assert issubclass(model, BaseModel)

        schema = model.model_json_schema()
        assert "anyOf" in schema["properties"]["x"]
        assert len(schema["properties"]["x"]["anyOf"]) == 2

    def test_function_with_optional(self):
        def optional_fn(x: Optional[int] = None, y: str = "default") -> None:
            pass

        model = pydantic_function_signature(optional_fn)
        assert issubclass(model, BaseModel)

        schema = model.model_json_schema()
        assert "x" in schema["properties"]
        assert "anyOf" in schema["properties"]["x"]
        assert len(schema["properties"]["x"]["anyOf"]) == 2
        assert "y" in schema["properties"]
        assert schema["properties"]["y"]["default"] == "default"

    def test_args_kwargs(self):
        def variadic_fn(*args: int, **kwargs: str) -> None:
            pass

        model = pydantic_function_signature(variadic_fn)
        assert issubclass(model, BaseModel)

        schema = model.model_json_schema()
        assert "args_list" in schema["properties"]
        assert "kwargs_dict" in schema["properties"]
        assert schema["properties"]["args_list"]["type"] == "array"
        assert schema["properties"]["kwargs_dict"]["type"] == "object"

    def test_class_method(self):
        class SampleClass:
            def method(self, x: int) -> None:
                pass

        instance = SampleClass()
        model = pydantic_function_signature(instance.method)
        assert issubclass(model, BaseModel)

        schema = model.model_json_schema()
        assert schema["title"] == "method"
        assert "x" in schema["properties"]
        assert "self" not in schema["properties"]

    def test_function_with_docstring(self):
        def documented_fn(x: int) -> None:
            """Test documentation."""
            pass

        model = pydantic_function_signature(documented_fn)
        assert issubclass(model, BaseModel)

        schema = model.model_json_schema()
        assert schema["description"] == "Test documentation."

    def test_function_without_types(self, caplog):
        def untyped_fn(x, y):
            pass

        caplog.set_level(logging.INFO, logger="yaaal.core.tool")

        _ = pydantic_function_signature(untyped_fn)
        logs = caplog.record_tuples

        assert any("No type annotation provided" in lg[2] for lg in logs)

    def test_function_with_missing_docstring(self, caplog):
        def undocumented_fn(x: int) -> None:
            pass

        caplog.set_level(logging.INFO, logger="yaaal.core.tool")

        model = pydantic_function_signature(undocumented_fn)
        schema = model.model_json_schema()
        assert "description" not in schema

        logs = caplog.record_tuples
        assert any("requires docstrings for viable signature" in lg[2] for lg in logs)

    def test_function_with_complex_types(self):
        class NestedModel(BaseModel):
            value: int

        def complex_fn(x: list[NestedModel], y: dict[str, int]) -> None:
            pass

        model = pydantic_function_signature(complex_fn)
        schema = model.model_json_schema()
        assert "NestedModel" in schema.get("$defs", {})
        assert schema["properties"]["x"]["type"] == "array"
        assert schema["properties"]["y"]["type"] == "object"

    def test_class_method_with_cls(self):
        class TestClass:
            @classmethod
            def class_method(cls, x: int) -> None:
                pass

        model = pydantic_function_signature(TestClass.class_method)
        schema = model.model_json_schema()
        assert "x" in schema["properties"]
        assert "cls" not in schema["properties"]


class TestToolDecorator:
    @pytest.fixture
    def add3(self):
        @tool
        def add3(x: int, y: int) -> int:
            return x + y + 3

        return add3

    @pytest.fixture
    def name3(self):
        @tool
        def name3(x: int, name: str) -> str:
            return f"{name}{x}3"

        return name3

    def test_tool_decorator_call(self, add3, name3):
        assert add3(1, 2) == 6  # 1 + 2 + 3
        assert name3(1, "bob") == "bob13"

    def test_tool_decorator_signature(self, add3, name3):
        assert hasattr(add3, "signature")
        model = add3.signature
        assert issubclass(model, BaseModel)

        schema = model.model_json_schema()
        assert schema["title"] == "add3"
        assert "x" in schema["properties"]
        assert schema["properties"]["x"]["type"] == "integer"
        assert "y" in schema["properties"]
        assert schema["properties"]["y"]["type"] == "integer"
        assert set(schema["required"]) == {"x", "y"}

        assert hasattr(name3, "signature")
        model = name3.signature
        assert issubclass(model, BaseModel)

        schema = model.model_json_schema()
        assert schema["title"] == "name3"
        assert "x" in schema["properties"]
        assert schema["properties"]["x"]["type"] == "integer"
        assert "name" in schema["properties"]
        assert schema["properties"]["name"]["type"] == "string"
        assert set(schema["required"]) == {"x", "name"}


class TestToolClass:
    def test_tool_without_return_type(self):
        def no_return(x: int):
            pass

        tool = Tool(no_return)
        assert tool.returns is None

    def test_tool_respond_as_tool_with_string(self):
        result = Tool.respond_as_tool("test_id", "test response")
        assert isinstance(result, ToolResultMessage)
        assert result.tool_call_id == "test_id"
        assert result.content == "test response"

    def test_tool_respond_as_tool_with_pydantic(self):
        class TestModel(BaseModel):
            value: str = "test"

        result = Tool.respond_as_tool("test_id", TestModel())
        assert isinstance(result, ToolResultMessage)
        assert result.tool_call_id == "test_id"
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

        tool = Tool(test_fn)
        assert tool.__name__ == "test_fn"
        assert tool.__doc__ == "Test function"


def test_validate_return_type_error():
    # Test that a tool returning a string cannot be converted to int.
    def fn(x: int) -> int:
        return "not an int"

    t = Tool(fn)
    with pytest.raises(ValueError):
        t(1)


def test_tool_with_none_return_should_fail():
    # Test that a tool returning None where int is expected raises a TypeError.
    def fn(x: int) -> int:
        return None

    t = Tool(fn)
    with pytest.raises(TypeError):
        t(1)


def test_tool_with_invalid_model_return():
    # Test that a tool returning an invalid dict for a BaseModel raises a ValidationError.
    from pydantic import BaseModel, ValidationError as PydanticValidationError

    class ExampleModel(BaseModel):
        value: int

    def fn(x: int) -> ExampleModel:
        # Return a dict with a wrong type for 'value'
        return {"value": "not an int"}

    t = Tool(fn, returns=ExampleModel)
    with pytest.raises(PydanticValidationError):
        t(1)


def test_anthropic_pydantic_function_tool():
    class TestModel(BaseModel):
        """Test model description"""

        x: int
        y: str = "default"

    result = anthropic_pydantic_function_tool(TestModel)
    assert result["name"] == "TestModel"
    assert result["description"] == "Test model description"
    assert set(result["input_schema"]["properties"]) == {"x", "y"}
    assert set(result["input_schema"]["required"]) == {"x", "y"}
