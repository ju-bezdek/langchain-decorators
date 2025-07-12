import pytest
import base64
from typing import List
from langchain_decorators.schema import MessageAttachment


class TestMessageAttachment:
    """Test cases for MessageAttachment class."""

    def test_attachment_creation_with_url(self):
        """Test creating attachment with URL input."""
        attachment = MessageAttachment(
            type="image",
            input="https://example.com/image.jpg",
            mime_type="image/jpeg"
        )
        
        assert attachment.type == "image"
        assert attachment.source_type == "url"
        assert attachment.source == {"url": "https://example.com/image.jpg"}
        assert attachment.mime_type == "image/jpeg"

    def test_attachment_creation_with_bytes(self):
        """Test creating attachment with bytes input."""
        test_bytes = b"fake image data"
        attachment = MessageAttachment(
            type="image",
            input=test_bytes,
            mime_type="image/png"
        )
        
        expected_b64 = base64.b64encode(test_bytes).decode('utf-8')
        
        assert attachment.type == "image"
        assert attachment.source_type == "base64"
        assert attachment.source == {
            "base64": expected_b64,
            "mime_type": "image/png"
        }

    def test_attachment_creation_with_base64_string(self):
        """Test creating attachment with base64 string input."""
        b64_string = "iVBORw0KGgoAAAANSUhEUgAA"
        attachment = MessageAttachment(
            type="image",
            input=b64_string,
            source_type="base64",
            mime_type="image/png"
        )
        
        assert attachment.source_type == "base64"
        assert attachment.source == {
            "base64": b64_string,
            "mime_type": "image/png"
        }

    def test_auto_detect_source_type_url(self):
        """Test auto-detection of URL source type."""
        attachment = MessageAttachment(
            type="image",
            input="https://example.com/image.jpg"
        )
        assert attachment.source_type == "url"

    def test_auto_detect_source_type_http(self):
        """Test auto-detection of HTTP URL source type."""
        attachment = MessageAttachment(
            type="image", 
            input="http://example.com/image.jpg"
        )
        assert attachment.source_type == "url"

    def test_auto_detect_source_type_base64_string(self):
        """Test auto-detection of base64 source type for strings."""
        attachment = MessageAttachment(
            type="image",
            input="not-a-url-string"
        )
        assert attachment.source_type == "base64"

    def test_auto_detect_source_type_bytes(self):
        """Test auto-detection of base64 source type for bytes."""
        attachment = MessageAttachment(
            type="image",
            input=b"image bytes"
        )
        assert attachment.source_type == "base64"

    def test_custom_source_override(self):
        """Test providing custom source dictionary."""
        custom_source = {
            "url": "https://custom.com/image.jpg",
            "custom_field": "value"
        }
        
        attachment = MessageAttachment(
            type="image",
            input="https://example.com/image.jpg",
            source=custom_source
        )
        
        assert attachment.source == custom_source

    def test_extra_metadata(self):
        """Test extra metadata field."""
        extra_data = {"metadata": "test", "custom": "value"}
        attachment = MessageAttachment(
            type="image",
            input="https://example.com/image.jpg",
            extra=extra_data
        )
        
        assert attachment.extra == extra_data

    def test_file_name_field(self):
        """Test file_name field."""
        attachment = MessageAttachment(
            type="pdf",
            input="https://example.com/document.pdf",
            file_name="report.pdf"
        )
        
        assert attachment.file_name == "report.pdf"

    def test_invalid_input_type(self):
        """Test validation of invalid input types."""
        with pytest.raises(ValueError, match="input must be str or bytes"):
            MessageAttachment(
                type="image",
                input=123  # Invalid type
            )

    def test_different_attachment_types(self):
        """Test different attachment types."""
        types_to_test = ["image", "file", "pdf", "audio"]
        
        for attachment_type in types_to_test:
            attachment = MessageAttachment(
                type=attachment_type,
                input="https://example.com/file"
            )
            assert attachment.type == attachment_type


class TestAttachmentUsageExamples:
    """Test cases for attachment usage examples from documentation."""

    def test_simple_image_url_example(self):
        """Test the simple image URL example."""
        attachment = MessageAttachment(
            type="image",
            input="https://example.com/image.jpg",
            mime_type="image/jpeg"
        )
        
        assert attachment.type == "image"
        assert attachment.source_type == "url"
        assert attachment.mime_type == "image/jpeg"

    def test_base64_image_bytes_example(self):
        """Test base64 image with bytes example."""
        fake_image_bytes = b"fake image data"
        
        attachment = MessageAttachment(
            type="image",
            input=fake_image_bytes,
            mime_type="image/png",
            file_name="screenshot.png"
        )
        
        assert attachment.type == "image"
        assert attachment.source_type == "base64"
        assert attachment.mime_type == "image/png"
        assert attachment.file_name == "screenshot.png"

    def test_base64_string_example(self):
        """Test base64 string example."""
        b64_data = "iVBORw0KGgoAAAANSUhEUgAA"
        
        attachment = MessageAttachment(
            type="image",
            input=b64_data,
            source_type="base64",
            mime_type="image/png"
        )
        
        assert attachment.source_type == "base64"
        assert attachment.source["base64"] == b64_data

    def test_pdf_attachment_example(self):
        """Test PDF attachment example."""
        attachment = MessageAttachment(
            type="pdf",
            input="https://example.com/document.pdf",
            file_name="report.pdf",
            mime_type="application/pdf"
        )
        
        assert attachment.type == "pdf"
        assert attachment.file_name == "report.pdf"
        assert attachment.mime_type == "application/pdf"

    def test_audio_attachment_example(self):
        """Test audio attachment example."""
        fake_audio_bytes = b"fake audio data"
        
        attachment = MessageAttachment(
            type="audio",
            input=fake_audio_bytes,
            mime_type="audio/wav",
            file_name="recording.wav"
        )
        
        assert attachment.type == "audio"
        assert attachment.mime_type == "audio/wav"
        assert attachment.file_name == "recording.wav"

    def test_custom_source_example(self):
        """Test custom source configuration example."""
        custom_source = {
            "base64": "base64data...",
            "mime_type": "image/jpeg",
            "custom_field": "value"
        }
        
        attachment = MessageAttachment(
            type="image",
            input="base64data...",
            source=custom_source,
            extra={"metadata": "additional info"}
        )
        
        assert attachment.source == custom_source
        assert attachment.extra == {"metadata": "additional info"}

    def test_multiple_attachments_list(self):
        """Test creating list of attachments."""
        attachments = [
            MessageAttachment(type="image", input="https://example.com/img1.jpg"),
            MessageAttachment(type="image", input="https://example.com/img2.jpg")
        ]
        
        assert len(attachments) == 2
        assert all(att.type == "image" for att in attachments)
        assert all(att.source_type == "url" for att in attachments)


if __name__ == "__main__":
    pytest.main([__file__])