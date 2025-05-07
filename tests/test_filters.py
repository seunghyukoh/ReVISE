import pytest

from revise.filters import RegexFilter


@pytest.mark.unit
class TestRegexFilter:
    def test_run_single(self):
        filter = RegexFilter(pattern=r"test\d")
        assert filter.run_single("test1 test2") == "test1"

    def test_run_single_with_group(self):
        filter_full = RegexFilter(pattern=r"(test\d)", select_index=0)
        assert filter_full.run_single("test1 test2") == "test1"

        filter_first = RegexFilter(pattern=r"(test\d)", select_index=1)
        assert filter_first.run_single("test1 test2") == "test2"

    def test_run_single_with_timeout(self):
        filter = RegexFilter(pattern=r".*", timeout=0.0)
        assert filter.run_single("test") == "[timeout]"

    def test_run_single_with_no_match(self):
        filter = RegexFilter(pattern=r"test\d")
        assert filter.run_single("test") == "[no match]"

    def test_run_single_with_error(self):
        filter = RegexFilter(pattern=123)
        assert filter.run_single("test") == "[error]"

    def test_run(self):
        filter = RegexFilter(pattern=r"test\d")
        assert filter.run(["test1", "test2"]) == ["test1", "test2"]

    def test_run_with_timeout(self):
        filter = RegexFilter(pattern=r".*", timeout=0.0)
        assert filter.run(["test1", "test2"]) == ["[timeout]", "[timeout]"]

    def test_run_with_no_match(self):
        filter = RegexFilter(pattern=r"test\d")
        assert filter.run(["test", "test"]) == ["[no match]", "[no match]"]

    def test_run_with_error(self):
        filter = RegexFilter(pattern=123)
        assert filter.run(["test1", "test2"]) == ["[error]", "[error]"]

    # Math
    def test_run_single_with_math(self):
        correct_response = """Natalia sold 48/2 = 24 clips in May.
Natalia sold 48+24 = 72 clips altogether in April and May.
#### 72"""

        filter = RegexFilter(pattern=r"(-?[$0-9.,]{2,})|(-?[0-9]+)", select_index=-1)
        assert filter.run_single(correct_response) == "72"

    # Code
    def test_run_single_with_code(self):
        correct_response = """
        ```python
        print("Hello, world!")
        ```
        """

        filter = RegexFilter(pattern=r"```python\s*((.|\s)*?)```", select_index=-1)
        assert filter.run_single(correct_response) == 'print("Hello, world!")'

    def test_run_single_with_long_code(self):
        correct_response = """To solve the problem of finding the first repeated character in a given string, we can utilize a set to keep track of the characters we have already encountered as we iterate through the string. The steps we will follow are:

1. Initialize an empty set to store characters that we've seen.
2. Loop through each character in the string.
3. For each character, check if it is already in the set:
- If it is, that means it is the first repeated character, and we return that character.
- If it is not, we add it to the set.
4. If we finish looping through the string without finding any repeated characters, we return `None`.

This approach is efficient because checking membership in a set is on average O(1), allowing us to efficiently determine if a character has been seen before.

Now let's implement this logic in code.

The answer is,
```python
def first_repeated_char(s):
    seen = set()
    for char in s:
        if char in seen:
            return char
        seen.add(char)
    return None
```"""

        filter = RegexFilter(pattern=r"```python\s*((.|\s)*?)```", select_index=-1)
        assert (
            filter.run_single(correct_response)
            == """def first_repeated_char(s):
    seen = set()
    for char in s:
        if char in seen:
            return char
        seen.add(char)
    return None"""
        )
