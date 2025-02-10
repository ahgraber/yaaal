from __future__ import annotations

import logging
from typing import Annotated, Any, Union

from pydantic import (
    ValidationError,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    WrapValidator,
)
from pydantic_core import PydanticCustomError
from typing_extensions import TypeAliasType  # TODO: import from typing when drop support for 3.11

logger = logging.getLogger(__name__)


def json_simple_error_validator(value: Any, handler: ValidatorFunctionWrapHandler, _info: ValidationInfo) -> Any:
    """Simplify the error message to avoid a gross error stemming from exhaustive checking of all union options."""
    try:
        return handler(value)
    except ValidationError as e:
        raise PydanticCustomError("invalid_json", "Input is not valid json") from e


JSONValue = Union[
    str,  # JSON string
    int,  # JSON number (integer)
    float,  # JSON number (float)
    bool,  # JSON boolean
    None,  # JSON null
]
JSON = TypeAliasType(
    "JSON",
    Annotated[
        Union[dict[str, "JSON"], list["JSON"], JSONValue],
        WrapValidator(json_simple_error_validator),
    ],
)
# JSONValidator = TypeAdapter[JSON]
# JSONModel = RootModel[JSON]
