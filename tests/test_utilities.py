import pytest

from yaaal.utilities import to_snake_case


class TestSnakeCase:
    def test_snake_case(self):
        assert to_snake_case("hello_world") == "hello_world"

    def test_with_spaces(self):
        assert to_snake_case("hello world") == "hello_world"

    def test_camel_case(self):
        assert to_snake_case("helloWorld") == "hello_world"

    def test_pascal_case(self):
        assert to_snake_case("HelloWorld") == "hello_world"

    def test_kebab_case(self):
        assert to_snake_case("hello-world") == "hello_world"

    def test_screaming_kebabs(self):
        assert to_snake_case("HELLO-WORLD") == "hello_world"

    def test_acronyms(self):
        assert to_snake_case("HTTPHeader") == "http_header"

    def test_mixed_case_with_acronyms(self):
        assert to_snake_case("helloWorldHTTPHeader") == "hello_world_http_header"

    def test_with_numbers(self):
        assert to_snake_case("test123Case") == "test123_case"

    def test_with_leading_trailing_spaces(self):
        assert to_snake_case("  hello world  ") == "hello_world"

    def test_to_snake_case_with_multiple_spaces(self):
        assert to_snake_case("hello   world") == "hello_world"

    def test_to_snake_case_with_multiple_hyphens(self):
        assert to_snake_case("hello---world") == "hello_world"
