# ruff: NOQA: N806
import types
from typing import Annotated, Any, Optional, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field
import pytest
import typing_extensions

from yaaal.types.utils import (
    TypeMergeError,
    _is_subclass_safe,
    _resolve_collection_types,
    _resolve_inner_types,
    _resolve_type_conflict,
    _resolve_union_types,
    _resolve_union_types_direct,
    get_union_args,
    merge_models,
    origin_is_union,
)


class TestResolveTypeConflict:
    def test_identical_types(self):
        assert _resolve_type_conflict(int, int) is int

    def test_any_types(self):
        assert _resolve_type_conflict(int, Any) is Any
        assert _resolve_type_conflict(Any, str) is Any

    def test_optional_type(self):
        res = _resolve_type_conflict(int, Optional[int])
        assert res == Optional[int] or res == (int | type(None))


class TestResolveInnerTypes:
    def test_basic(self):
        class A:
            pass

        class B(A):
            pass

        res = _resolve_inner_types(B, A)
        assert res == A
        res2 = _resolve_inner_types(A, B)
        assert res2 == A

    def test_numeric(self):
        res = _resolve_inner_types(int, float)
        assert res is float
        res2 = _resolve_inner_types(float, int)
        assert res2 is float

    def test_bool_int(self):
        res = _resolve_inner_types(bool, int)
        assert res is int
        res2 = _resolve_inner_types(int, bool)
        assert res2 is int

    def test_string_types(self):
        # str vs bytes should not merge
        with pytest.raises(TypeMergeError):
            _resolve_inner_types(str, bytes)

        # str vs str subclass should resolve to str
        class CustomString(str):
            pass

        assert _resolve_inner_types(CustomString, str) is str

    def test_abstract_types(self):
        from collections.abc import MutableSequence, Sequence

        # Abstract base class relationships
        assert _resolve_inner_types(list, MutableSequence) == MutableSequence
        assert _resolve_inner_types(tuple, Sequence) == Sequence

    def test_custom_class_hierarchy(self):
        class Base:
            pass

        class Middle(Base):
            pass

        class Leaf(Middle):
            pass

        # Test all combinations of hierarchy
        assert _resolve_inner_types(Leaf, Middle) is Middle
        assert _resolve_inner_types(Leaf, Base) is Base
        assert _resolve_inner_types(Middle, Base) is Base
        assert _resolve_inner_types(Base, Middle) is Base

    def test_protocol_types(self):
        from typing import Protocol

        class Sizeable(Protocol):
            def __len__(self) -> int: ...

        # list implements Sizeable protocol but isn't a subclass
        with pytest.raises(TypeMergeError):
            _resolve_inner_types(list, Sizeable)

    def test_incompatible_types(self):
        # Different builtin types should not merge
        with pytest.raises(TypeMergeError):
            _resolve_inner_types(list, dict)

        with pytest.raises(TypeMergeError):
            _resolve_inner_types(str, int)

        class A:
            pass

        class B:
            pass

        # Unrelated custom classes should not merge
        with pytest.raises(TypeMergeError):
            _resolve_inner_types(A, B)


class TestResolveCollectionTypes:
    def test_list(self):
        t1 = list[int]
        t2 = list[int]
        res = _resolve_collection_types(t1, t2)
        assert res == list[int]

    def test_dict(self):
        t1 = dict[str, int]
        t2 = dict[str, int]
        res = _resolve_collection_types(t1, t2)
        assert res == dict[str, int]

    def test_tuple(self):
        t1 = tuple[int, str]
        t2 = tuple[int, str]
        res = _resolve_collection_types(t1, t2)
        assert get_args(res) == get_args(t1)
        t3 = tuple[int, ...]
        t4 = tuple[int, ...]
        res2 = _resolve_collection_types(t3, t4)
        assert res2 == t3

    def test_tuple_fixed_length(self):
        t1 = tuple[int, float]
        t2 = tuple[int, float]
        res = _resolve_collection_types(t1, t2)
        # Expected tuple with two resolved types
        assert get_args(res) == get_args(t1)

    def test_tuple_variable_length_fallback(self):
        # When tuple lengths differ, fallback to variable-length tuple
        t1 = tuple[int, str]
        t2 = tuple[int]
        res = _resolve_collection_types(t1, t2)
        assert res == tuple[Any, ...]  # Fallback behavior

    def test_mismatched_collection_types_raise_error(self):
        # Using different origins should raise TypeMergeError
        t1 = list[int]
        t2 = set[int]
        with pytest.raises(TypeMergeError, match="Cannot merge different collection types"):
            _resolve_collection_types(t1, t2)

    def test_incompatible_types_raise_error(self):
        # Trying to merge str and int should raise TypeMergeError
        with pytest.raises(TypeMergeError, match="Cannot merge incompatible types"):
            _resolve_inner_types(str, int)

    def test_unsupported_collection_raises_error(self):
        # Using an unsupported collection type should raise TypeMergeError
        class CustomCollection:
            pass

        with pytest.raises(TypeMergeError):  # , match="Unsupported collection type"):
            _resolve_collection_types(CustomCollection(), CustomCollection())


class TestUnionTypes:
    def test_direct_simple(self):
        res = _resolve_union_types_direct((int,), (str,))
        expected = Union[int, str]
        assert set(get_args(res)) == set(get_args(expected))

    def test_direct_disjoint(self):
        # For disjoint sets: (int, float) and (str, bool)
        # In Python, bool is a subclass of int so it should be dropped.
        res = _resolve_union_types_direct((int, float), (str, bool))
        expected_types = {int, float, str}
        assert set(get_args(res)) == expected_types

    def test_direct_overlapping(self):
        # Overlapping unions should result in a union with distinct types.
        res = _resolve_union_types_direct((int, str), (str, int))
        expected = Union[int, str]
        assert set(get_args(res)) == set(get_args(expected))

    def test_direct_redundant(self):
        # With redundant types, the function should return the single type.
        res = _resolve_union_types_direct((bool,), (int,))
        # Since bool is a subclass of int, expected result is int.
        assert res is int

    def test_union_single_argument(self):
        # If both unions only yield one unique type then the result should be that type directly.
        res = _resolve_union_types_direct((int,), (int,))
        assert res is int


class TestGetUnionArgs:
    def test_non_union(self):
        res = get_union_args(int)
        assert res == (int,)

    def test_union(self):
        res = get_union_args(Optional[int])
        assert set(res) == {int, type(None)}


class TestOriginIsUnion:
    def test_origin(self):
        union_type = Union[int, str]
        assert origin_is_union(get_origin(union_type))
        union_operator = int | str
        assert origin_is_union(get_origin(union_operator))


class TestMergeModels:
    def test_merge_simple_models(self):
        class ModelA(BaseModel):
            str_field: str
            int_field: int = 42

        class ModelB(BaseModel):
            int_field: int = 10
            float_field: float

        MergedModel = merge_models(ModelA, ModelB, name="TestMerge")

        # Check merged model structure
        assert MergedModel.__name__ == "TestMerge"
        assert set(MergedModel.model_fields.keys()) == {"str_field", "int_field", "float_field"}

        # Check field types
        assert MergedModel.model_fields["str_field"].annotation is str
        assert MergedModel.model_fields["int_field"].annotation is int
        assert MergedModel.model_fields["float_field"].annotation is float

        # Check defaults
        assert MergedModel.model_fields["int_field"].default == 42  # First model's default preserved

    def test_merge_with_type_conflicts(self):
        class ModelA(BaseModel):
            field1: int

        class ModelB(BaseModel):
            field1: float

        class ModelC(BaseModel):
            field1: bool

        MergedModelAB = merge_models(ModelA, ModelB)
        # float is the resolved type for int vs float
        assert MergedModelAB.model_fields["field1"].annotation is float

        MergedModelAC = merge_models(ModelA, ModelC)
        # int is the resolved type for int vs bool
        assert MergedModelAC.model_fields["field1"].annotation is int

        with pytest.raises(TypeMergeError):
            merge_models(ModelB, ModelC)

    def test_merge_with_defaults(self):
        class ModelA(BaseModel):
            field1: int

        class ModelB(BaseModel):
            field1: int = 42

        class ModelC(BaseModel):
            field1: int = 10

        MergedModel = merge_models(ModelA, ModelB, ModelC)
        # First non-None default should be preserved
        assert MergedModel.model_fields["field1"].default == 42

    def test_merge_with_descriptions(self):
        class ModelA(BaseModel):
            field1: str

        class ModelB(BaseModel):
            field1: str = Field(description="First description")

        class ModelC(BaseModel):
            field1: str = Field(description="Second description")

        MergedModel = merge_models(ModelA, ModelB, ModelC)
        # First non-None description should be preserved
        assert MergedModel.model_fields["field1"].description == "First description"

    def test_merge_with_required_fields(self):
        class ModelA(BaseModel):
            field1: str | None = None

        class ModelB(BaseModel):
            field1: str

        MergedModel = merge_models(ModelA, ModelB)
        # Field should be required if required in any model
        assert MergedModel.model_fields["field1"].is_required()

    def test_merge_with_config(self):
        config = ConfigDict(extra="forbid")

        class ModelA(BaseModel):
            field1: str

        MergedModel = merge_models(ModelA, config=config)
        assert MergedModel.model_config["extra"] == "forbid"

    def test_merge_empty_models_raises(self):
        with pytest.raises(ValueError, match="At least one model must be provided"):
            merge_models()

    def test_merge_incompatible_types_raises(self):
        class ModelA(BaseModel):
            field1: str

        class ModelB(BaseModel):
            field1: int  # Incompatible with str

        with pytest.raises(TypeMergeError):
            merge_models(ModelA, ModelB)

    def test_merge_with_nested_models(self):
        class NestedModel(BaseModel):
            nested_field: str

        class ModelA(BaseModel):
            field1: NestedModel

        class ModelB(BaseModel):
            field2: NestedModel

        MergedModel = merge_models(ModelA, ModelB)
        assert MergedModel.model_fields["field1"].annotation == NestedModel
        assert MergedModel.model_fields["field2"].annotation == NestedModel
