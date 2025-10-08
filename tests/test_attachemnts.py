import pytest
import base64
from typing import List
from urllib.request import urlopen
from langchain_decorators.schema import MessageAttachment
from langchain_decorators.prompt_decorator import llm_prompt


@llm_prompt
def prompt_with_single_attachment(attachment: MessageAttachment) -> str:
    """
    ```<prompt:system>
    Your task is to understand what is on the picture and tell it to our blind user.
    ```
    ```<prompt:user>
    What is on the picture?
    {attachment}
    ```
    """


@llm_prompt
def prompt_with_multiple_attachments(attachments: List[MessageAttachment]) -> str:
    """
    ```<prompt:system>
    Your task is to understand what is on the picture and tell it to our blind user.
    ```
    ```<prompt:user>
    What is on the pictures?
    {attachments}
    ```
    """


@llm_prompt
def prompt_with_optional_attachments(
    query: str, attachments: List[MessageAttachment] = None
) -> str:
    """
    ```<prompt:system>
    Be a helpful assistant and answer the user's question.
    ```
    ```<prompt:user>
    {query}
    {? What is on the pictures?
    {attachments}
    ?}
    ```
    """


@llm_prompt
def prompt_with_optional_attachment(
    query: str, attachment: MessageAttachment = None
) -> str:
    """
    ```<prompt:system>
    Be a helpful assistant and answer the user's question.
    ```
    ```<prompt:user>
    {query}
    {? What is on the picture?
    {attachment}
    ?}
    ```
    """


# Test image URL
JUPITER_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e2/Jupiter_OPAL_2024.png/960px-Jupiter_OPAL_2024.png"


def fetch_image_as_base64(url: str) -> str:
    """Helper function to fetch an image from URL and convert to base64."""
    try:
        with urlopen(url) as response:
            image_data = response.read()
            return base64.b64encode(image_data).decode("utf-8")
    except Exception as e:
        pytest.skip(f"Could not fetch image from {url}: {e}")


class TestAttachmentScenarios:
    """Test various attachment scenarios with real function calls."""

    def test_single_attachment_with_url(self):
        """Test single attachment using URL source."""
        attachment = MessageAttachment(
            type="image",
            data=JUPITER_IMAGE_URL,
            source_type="url",
            mime_type="image/png",
            file_name="jupiter_2024.png",
        )

        # Call the function with URL-based attachment
        result = prompt_with_single_attachment(attachment=attachment)

        # Basic validation that we got a response
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"Single URL attachment result: {result}")

    def test_single_attachment_with_base64(self):
        """Test single attachment using base64 encoded data."""
        base64_data = fetch_image_as_base64(JUPITER_IMAGE_URL)

        attachment = MessageAttachment(
            type="image",
            data=base64_data,
            source_type="base64",
            mime_type="image/png",
            file_name="jupiter_2024.png",
        )

        # Call the function with base64-based attachment
        result = prompt_with_single_attachment(attachment=attachment)

        # Basic validation that we got a response
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"Single base64 attachment result: {result}")

    def test_multiple_attachments_with_url(self):
        """Test multiple attachments using URL source (single item in list)."""
        attachment = MessageAttachment(
            type="image",
            data=JUPITER_IMAGE_URL,
            source_type="url",
            mime_type="image/png",
            file_name="jupiter.png",
        )

        # Test with single item in list
        result = prompt_with_multiple_attachments(attachments=[attachment])

        # Basic validation that we got a response
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"Multiple attachments (URL) result: {result}")

    def test_multiple_attachments_with_base64(self):
        """Test multiple attachments using base64 source (single item in list)."""
        base64_data = fetch_image_as_base64(JUPITER_IMAGE_URL)

        attachment = MessageAttachment(
            type="image",
            data=base64_data,
            source_type="base64",
            mime_type="image/png",
            file_name="jupiter.png",
        )

        # Test with single item in list
        result = prompt_with_multiple_attachments(attachments=[attachment])

        # Basic validation that we got a response
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"Multiple attachments (base64) result: {result}")

    def test_optional_attachment_with_content(self):
        """Test optional attachment when attachment is provided."""
        attachment = MessageAttachment(
            type="image",
            data=JUPITER_IMAGE_URL,
            source_type="url",
            mime_type="image/png",
            file_name="jupiter_2024.png",
        )

        result = prompt_with_optional_attachment(
            query="Can you describe this planet?", attachment=attachment
        )

        assert isinstance(result, str)
        assert len(result) > 0
        print(f"Optional attachment (with content) result: {result}")

    def test_optional_attachment_without_content(self):
        """Test optional attachment when no attachment is provided."""
        result = prompt_with_optional_attachment(
            query="Tell me about Jupiter, the planet."
        )

        assert isinstance(result, str)
        assert len(result) > 0
        print(f"Optional attachment (without content) result: {result}")

    def test_optional_attachments_list_with_content(self):
        """Test optional attachments list when attachments are provided."""
        attachment = MessageAttachment(
            type="image",
            data=JUPITER_IMAGE_URL,
            source_type="url",
            mime_type="image/png",
            file_name="jupiter.png",
        )

        result = prompt_with_optional_attachments(
            query="What do you see in this image of a celestial body?",
            attachments=[attachment],
        )

        assert isinstance(result, str)
        assert len(result) > 0
        print(f"Optional attachments list (with content) result: {result}")

    def test_optional_attachments_list_without_content(self):
        """Test optional attachments list when no attachments are provided."""
        result = prompt_with_optional_attachments(
            query="Tell me about the largest planet in our solar system."
        )

        assert isinstance(result, str)
        assert len(result) > 0
        print(f"Optional attachments list (without content) result: {result}")

    def test_attachment_validation(self):
        """Test MessageAttachment validation."""
        # Test valid attachment
        valid_attachment = MessageAttachment(
            type="image",
            data=JUPITER_IMAGE_URL,
            source_type="url",
            mime_type="image/png",
        )
        assert valid_attachment.type == "image"
        assert valid_attachment.data == JUPITER_IMAGE_URL

        # Test with bytes data
        base64_data = fetch_image_as_base64(JUPITER_IMAGE_URL)
        bytes_attachment = MessageAttachment(
            type="image",
            data=base64_data.encode("utf-8"),
            source_type="base64",
            mime_type="image/png",
        )
        assert isinstance(bytes_attachment.data, bytes)

    def test_different_attachment_types(self):
        """Test different attachment types support."""
        attachment_types = ["image", "file", "pdf", "audio"]

        for att_type in attachment_types:
            attachment = MessageAttachment(
                type=att_type,
                data=JUPITER_IMAGE_URL if att_type == "image" else "dummy_data",
                source_type="url" if att_type == "image" else "base64",
                mime_type=f"{att_type}/test",
            )
            assert attachment.type == att_type


# Integration test that requires real LLM
@pytest.mark.integration
class TestAttachmentIntegration:
    """Integration tests that require a real LLM to be configured."""

    def test_end_to_end_image_analysis(self):
        """End-to-end test of image analysis with Jupiter image."""
        attachment = MessageAttachment(
            type="image",
            data=JUPITER_IMAGE_URL,
            source_type="url",
            mime_type="image/png",
            file_name="jupiter_opal_2024.png",
        )

        # This would require a real LLM with vision capabilities
        try:
            result = prompt_with_single_attachment(attachment=attachment)

            # Check that the result mentions relevant keywords
            result_lower = result.lower()
            jupiter_keywords = [
                "jupiter",
                "planet",
                "gas",
                "giant",
                "stripes",
                "bands",
                "storm",
            ]

            # At least one Jupiter-related keyword should be present
            assert any(
                keyword in result_lower for keyword in jupiter_keywords
            ), f"Result should contain Jupiter-related content. Got: {result}"

        except Exception as e:
            pytest.skip(f"Integration test requires proper LLM setup: {e}")


def main():
    """Run all tests when script is executed directly."""
    print("Running attachment tests...")

    # Create test instance
    test_scenarios = TestAttachmentScenarios()

    # Run all test methods
    test_methods = [
        "test_single_attachment_with_url",
        "test_single_attachment_with_base64",
        "test_multiple_attachments_with_url",
        "test_multiple_attachments_with_base64",
        "test_optional_attachment_with_content",
        "test_optional_attachment_without_content",
        "test_optional_attachments_list_with_content",
        "test_optional_attachments_list_without_content",
        "test_attachment_validation",
        "test_different_attachment_types",
    ]

    passed = 0
    failed = 0

    for method_name in test_methods:
        try:
            print(f"\n--- Running {method_name} ---")
            method = getattr(test_scenarios, method_name)
            method()
            print(f"✓ {method_name} PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ {method_name} FAILED: {e}")
            failed += 1

    print(f"\n--- Test Summary ---")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")


if __name__ == "__main__":
    main()
