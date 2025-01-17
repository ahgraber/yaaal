from pydantic import ValidationError
import pytest

from yaaal.utilities.parse import (
    check_matched_pairs,
    extract_json,
)


class TestParensAreMatched:
    def test_balanced(self):
        testcases = ["abc", "(abc)a(b)c", "a(b)c(d)e", "()()()()", "(((())))"]

        for test in testcases:
            assert check_matched_pairs(test)

    def test_unbalanced(self):
        testcases = ["(abc", "a(b)c)", "a(b)c(d)e)", "(()", "())"]

        for test in testcases:
            assert not check_matched_pairs(test)

    def test_brackets(self):
        testcases = ["abc", "[abc]", "a[b]c", "a[b]c[d]e", "[][][]", "[[[]]]"]

        for test in testcases:
            assert check_matched_pairs(test, "[", "]")

    def test_braces(self):
        testcases = ["abc", "{abc}", "a{b}c", "a{b}c{d}e", "{}{}{}", "{{{}}}"]

        for test in testcases:
            assert check_matched_pairs(test, "{", "}")


class TestExtractJson:
    prefix = "Here's the generated abstract conceptual question in the requested JSON format: "
    suffix = "Would you like me to explain in more detail?"
    object = """{"key": "value"}"""
    array = """[1, 2, 3]"""
    nested = """{"outer": {"inner": [1, 2, 3]}}"""

    test_cases = [
        (object, object),
        (array, array),
        (nested, nested),
        (prefix + object, object),
        (object + suffix, object),
        (prefix + object + suffix, object),
        (prefix + array, array),
        (array + suffix, array),
        (prefix + array + suffix, array),
        (prefix + nested, nested),
        (nested + suffix, nested),
        (prefix + nested + suffix, nested),
        (object + array + nested, object),
        (nested + object + array, nested),
    ]

    @pytest.mark.parametrize("text, expected", test_cases)
    def test_extract_json(self, text, expected):
        assert extract_json(text) == expected

    def test_extract_empty_array(self):
        text = "Here is an empty array: [] and some text."
        expected = "[]"
        assert extract_json(text) == expected

    def test_extract_empty_object(self):
        text = "Here is an empty object: {} and more text."
        expected = "{}"
        assert extract_json(text) == expected

    def test_extract_incomplete_json(self):
        text = 'Not complete: {"key": "value", "array": [1, 2, 3'
        expected = 'Not complete: {"key": "value", "array": [1, 2, 3'
        assert extract_json(text) == expected

    def test_markdown_json(self):
        text = """
        ```python
        import json

        def modify_query(input_data):
            query = input_data["query"]
            style = input_data["style"]
            length = input_data["length"]

            if style == "Poor grammar":
                # Poor grammar modifications (simplified for brevity)
                query = query.replace("How", "how")
                query = query.replace("do", "does")
                query = query.replace("terms of", "in terms of")
                query = query.replace("and", "")

            if length == "long":
                # Long text modifications (simplified for brevity)
                query += "?"

            return {
                "text": query
            }

        input_data = {
            "query": "How can the provided commands be used to manage and troubleshoot namespaces in a Kubernetes environment?",
            "style": "Poor grammar",
            "length": "long"
        }

        output = modify_query(input_data)
        print(json.dumps(output, indent=4))
        ```

        Output:
        ```json
        {"text": "how does the provided commands be used to manage and troubleshoot namespaces in a Kubernetes environment?"}
        ```
        This Python function `modify_query` takes an input dictionary with query, style, and length as keys. It applies modifications based on the specified style (Poor grammar) and length (long). The modified query is then returned as a JSON object.

        Note: This implementation is simplified for brevity and may not cover all possible edge cases or nuances of natural language processing.
        """
        expected = """{"text": "how does the provided commands be used to manage and troubleshoot namespaces in a Kubernetes environment?"}"""
        assert extract_json(text) == expected
