from typing import Optional, Union

from pydantic import BaseModel
import pytest

from yaaal.core.tools import function_schema_model, respond_as_tool, tool
from yaaal.types.core import ToolMessage


class TestFunctionSchema:
    def test_basic_function(self):
        def sample_fn(x: int, y: str) -> None:
            pass

        model = function_schema_model(sample_fn)
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

        model = function_schema_model(union_fn)
        assert issubclass(model, BaseModel)

        schema = model.model_json_schema()
        assert "anyOf" in schema["properties"]["x"]
        assert len(schema["properties"]["x"]["anyOf"]) == 2

    def test_function_with_uniontype(self):
        def union_fn(x: int | str) -> None:
            pass

        model = function_schema_model(union_fn)
        assert issubclass(model, BaseModel)

        schema = model.model_json_schema()
        assert "anyOf" in schema["properties"]["x"]
        assert len(schema["properties"]["x"]["anyOf"]) == 2

    def test_function_with_optional(self):
        def optional_fn(x: Optional[int] = None, y: str = "default") -> None:
            pass

        model = function_schema_model(optional_fn)
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

        model = function_schema_model(variadic_fn)
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
        model = function_schema_model(instance.method)
        assert issubclass(model, BaseModel)

        schema = model.model_json_schema()
        assert schema["title"] == "method"
        assert "x" in schema["properties"]
        assert "self" not in schema["properties"]

    def test_function_with_docstring(self):
        def documented_fn(x: int) -> None:
            """Test documentation."""
            pass

        model = function_schema_model(documented_fn)
        assert issubclass(model, BaseModel)

        schema = model.model_json_schema()
        assert schema["description"] == "Test documentation."

    def test_function_without_types(self):
        def untyped_fn(x, y):
            pass

        with pytest.raises(TypeError):
            function_schema_model(untyped_fn)


class ToolResponse:
    def test_tool_response(self):
        tm = respond_as_tool(tool_call_id="test_id", response=6)
        assert isinstance(tm, ToolMessage)
        assert tm.tool_call_id == "test_id"
        assert tm.content == "6"

        tm = respond_as_tool(tool_call_id="test_id", response="bob")
        assert isinstance(tm, ToolMessage)
        assert tm.tool_call_id == "test_id"
        assert tm.content == "bob"

    def test_tool_response_no_response(self):
        tm = respond_as_tool(tool_call_id="test_id")
        assert isinstance(tm, ToolMessage)
        assert tm.tool_call_id == "test_id"
        assert tm.content == "null"

        tm = respond_as_tool(tool_call_id="test_id", response=None)
        assert isinstance(tm, ToolMessage)
        assert tm.tool_call_id == "test_id"
        assert tm.content == "null"

    def test_tool_response_with_model(self):
        class SampleModel(BaseModel):
            x: int
            y: str

        @tool
        def sample_fn(x: int, y: str) -> SampleModel:
            return SampleModel(x=x, y=y)

        tm = respond_as_tool(tool_call_id="test_id", response=sample_fn(x=1, y="test"))
        assert isinstance(tm, ToolMessage)
        assert tm.tool_call_id == "test_id"
        assert tm.content == '{"x":1,"y":"test"}'

    def test_tool_response_with_invalid_tool_call_id(self):
        with pytest.raises(ValueError):
            respond_as_tool(response="test")

        with pytest.raises(ValueError):
            respond_as_tool(tool_call_id=None, response="test")


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
        model = add3.signature()
        assert issubclass(model, BaseModel)

        schema = model.model_json_schema()
        assert schema["title"] == "add3"
        assert "x" in schema["properties"]
        assert schema["properties"]["x"]["type"] == "integer"
        assert "y" in schema["properties"]
        assert schema["properties"]["y"]["type"] == "integer"
        assert set(schema["required"]) == {"x", "y"}

        assert hasattr(name3, "signature")
        model = name3.signature()
        assert issubclass(model, BaseModel)

        schema = model.model_json_schema()
        assert schema["title"] == "name3"
        assert "x" in schema["properties"]
        assert schema["properties"]["x"]["type"] == "integer"
        assert "name" in schema["properties"]
        assert schema["properties"]["name"]["type"] == "string"
        assert set(schema["required"]) == {"x", "name"}

    def test_tool_decorator_response(self, add3, name3):
        response = add3.tool_response(1, 2, tool_call_id="test_id")
        assert isinstance(response, ToolMessage)
        assert response.tool_call_id == "test_id"
        assert response.content == "6"

        response = name3.tool_response(tool_call_id="test_id", x=1, name="bob")
        assert isinstance(response, ToolMessage)
        assert response.tool_call_id == "test_id"
        assert response.content == "bob13"

    def test_tool_response_with_model(self):
        class SampleModel(BaseModel):
            x: int
            y: str

        @tool
        def sample_fn(x: int, y: str) -> SampleModel:
            return SampleModel(x=x, y=y)

        response = sample_fn.tool_response(tool_call_id="test_id", x=1, y="test")
        assert isinstance(response, ToolMessage)
        assert response.tool_call_id == "test_id"
        assert response.content == '{"x":1,"y":"test"}'

    def test_tool_response_with_invalid_tool_call_id(self, add3):
        with pytest.raises(ValueError):
            add3.tool_response(1, 2, tool_call_id=None)
