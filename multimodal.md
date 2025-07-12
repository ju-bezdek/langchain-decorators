# Multimodal Support with Attachments

LangChain Decorators supports multimodal messages through attachment arguments. This allows you to include images, files, audio, and other media types in your prompts.

## Basic Usage

### Simple Attachment Parameter

```python
from langchain_decorators import llm_prompt
from langchain_decorators.schema import MessageAttachment

@llm_prompt
def analyze_image(question: str, image: MessageAttachment):
    """
    ```<prompt:user>
    Please analyze this image and answer: {question}
    {image}
    ```
    """

# Usage
attachment = MessageAttachment(
    type="image",
    input="https://example.com/image.jpg",
    mime_type="image/jpeg"
)

result = analyze_image("What do you see?", image=attachment)
```

### Inline Attachment Usage

```python
@llm_prompt
def describe_with_context(context: str, image: MessageAttachment, follow_up: str):
    """
    ```<prompt:user>
    {context}
    {image}
    {follow_up}
    ```
    """

# This creates a message with mixed content:
# [
#   {"type": "text", "text": "Context here"},
#   {"type": "image", "source": {...}},
#   {"type": "text", "text": "Follow-up text"}
# ]
```

### Unused Attachments

If an attachment is passed but not used in the template, it will be automatically added to the last message:

```python
@llm_prompt
def simple_prompt(text: str, unused_image: MessageAttachment):
    """
    ```<prompt:user>
    {text}
    ```
    """
    
# The image will be appended to the user message automatically
```

### Multiple Attachments

```python
from typing import List

@llm_prompt
def compare_images(instruction: str, images: List[MessageAttachment]):
    """
    ```<prompt:user>
    {instruction}
    {images}
    ```
    """

# Usage
images = [
    MessageAttachment(type="image", input="https://example.com/img1.jpg"),
    MessageAttachment(type="image", input="https://example.com/img2.jpg")
]

result = compare_images("Compare these images:", images=images)
```

### Direct Message Passing

You can also pass entire messages using placeholders:

```python
@llm_prompt
def custom_message_flow(system_msg: str, user_message: dict):
    """
    ```<prompt:system>
    {system_msg}
    ```
    
    ```<prompt:placeholder>
    {user_message}
    ```
    """

# Usage with pre-built message
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Analyze this"},
        {"type": "image", "source": {"url": "https://example.com/image.jpg"}}
    ]
}

result = custom_message_flow("You are an analyst", user_message=message)
```

## Attachment Types

### Image Attachments

```python
# URL-based image
image_url = MessageAttachment(
    type="image",
    input="https://example.com/image.jpg",
    mime_type="image/jpeg"
)

# Base64-encoded image
with open("image.png", "rb") as f:
    image_bytes = f.read()

image_base64 = MessageAttachment(
    type="image",
    input=image_bytes,
    mime_type="image/png",
    file_name="screenshot.png"
)

# Base64 string
image_b64_string = MessageAttachment(
    type="image",
    input="iVBORw0KGgoAAAANSUhEUgAA...",  # base64 string
    source_type="base64",
    mime_type="image/png"
)
```

### File Attachments

```python
# PDF file
pdf_attachment = MessageAttachment(
    type="pdf",
    input="https://example.com/document.pdf",
    file_name="report.pdf",
    mime_type="application/pdf"
)

# Audio file
audio_attachment = MessageAttachment(
    type="audio",
    input=audio_bytes,
    mime_type="audio/wav",
    file_name="recording.wav"
)
```

## Advanced Features

### Custom Source Configuration

```python
attachment = MessageAttachment(
    type="image",
    input="base64data...",
    source={
        "base64": "base64data...",
        "mime_type": "image/jpeg",
        "custom_field": "value"
    },
    extra={"metadata": "additional info"}
)
```

### Auto-Detection

The `source_type` is automatically detected:
- `bytes` input → `base64`
- `str` starting with `http://` or `https://` → `url`
- Other `str` → `base64`

```python
# These are equivalent
attachment1 = MessageAttachment(type="image", input="https://example.com/img.jpg")
attachment2 = MessageAttachment(
    type="image", 
    input="https://example.com/img.jpg", 
    source_type="url"
)
```

## Error Handling

```python
try:
    attachment = MessageAttachment(
        type="image",
        input=123  # Invalid type
    )
except ValueError as e:
    print(f"Invalid input: {e}")
```

## Best Practices

1. **Specify MIME types** for better compatibility
2. **Use descriptive file names** when available
3. **Handle large files appropriately** (consider file size limits)
4. **Use URL sources** for large media when possible to avoid payload bloat
5. **Validate attachment types** match your model's capabilities

## Model Compatibility

Different language models support different attachment types:
- **GPT-4V**: Images
- **Claude 3**: Images, documents
- **Gemini Pro Vision**: Images

Always check your model's documentation for supported attachment types and formats.