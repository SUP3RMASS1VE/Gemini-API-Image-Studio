import json
import os
import time
import uuid
import tempfile
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
import base64
import mimetypes
from dotenv import load_dotenv
import io

from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

def save_binary_file(file_name, data):
    with open(file_name, "wb") as f:
        f.write(data)

def generate(text, file_name, model="gemini-2.0-flash-exp", progress=gr.Progress()):
    # Get API key from environment variables
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")
        
    # Initialize client using API key from environment variable
    client = genai.Client(api_key=api_key)
    
    progress(0, desc="Uploading file to Gemini")
    files = [client.files.upload(file=file_name)]
    
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(
                    file_uri=files[0].uri,
                    mime_type=files[0].mime_type,
                ),
                types.Part.from_text(text=text),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        response_modalities=["image", "text"],
        response_mime_type="text/plain",
    )

    text_response = ""
    image_path = None
    progress(0.2, desc="Waiting for Gemini's response")
    # Create a temporary file to potentially store image data.
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_path = tmp.name
        chunk_count = 0
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                continue
            chunk_count += 1
            progress(0.2 + min(0.6 * (chunk_count / 10), 0.6), desc="Receiving response from Gemini")
            candidate = chunk.candidates[0].content.parts[0]
            # Check for inline image data
            if candidate.inline_data:
                progress(0.9, desc="Image data received, saving...")
                save_binary_file(temp_path, candidate.inline_data.data)
                image_path = temp_path
                # If an image is found, we assume that is the desired output.
                break
            else:
                # Accumulate text response if no inline_data is present.
                text_response += chunk.text + "\n"
    
    progress(1.0, desc="Processing complete")
    del files
    return image_path, text_response

def process_image_and_prompt(composite_pil, prompt, progress=gr.Progress()):
    try:
        # Save the composite image to a temporary file
        progress(0.1, desc="Saving uploaded image")
        # Convert to RGB if image is RGBA
        if composite_pil.mode == "RGBA":
            composite_pil = composite_pil.convert("RGB")
            
        # Save the input image to a temporary file first
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            composite_pil.save(tmp.name, format="JPEG", quality=95)
            tmp_path = tmp.name
        
        # Generate the image
        progress(0.2, desc="Starting Gemini AI generation")
        image_path, text_response = generate(text=prompt, file_name=tmp_path, model="gemini-2.0-flash-exp", progress=progress)
        
        if image_path:
            # Load and process the result
            progress(0.9, desc="Processing result")
            result_img = Image.open(image_path)
            if result_img.mode == "RGBA":
                result_img = result_img.convert("RGB")
            
            # Create a copy of the image for display
            display_img = result_img.copy()
            
            # Clean up temporary files
            try:
                os.unlink(tmp_path)
                os.unlink(image_path)
            except:
                pass
            
            # Return the image for display and download
            progress(1.0, desc="Complete!")
            return [display_img], "", result_img  # Return PIL Image for both display and download
        else:
            # Clean up temporary input file
            try:
                os.unlink(tmp_path)
            except:
                pass
                
            progress(1.0, desc="Complete!")
            return None, text_response, None
    except Exception as e:
        # Clean up temporary files in case of error
        try:
            os.unlink(tmp_path)
        except:
            pass
        raise gr.Error(f"Error: {e}", duration=5)

def update_env_file(api_key):
    """
    Updates the .env file with the provided API key.
    If the file doesn't exist, it creates it.
    """
    env_file_path = ".env"
    try:
        with open(env_file_path, "w") as f:
            f.write(f"GEMINI_API_KEY={api_key}\n")
        return "API key saved successfully!"
    except Exception as e:
        return f"Error saving API key: {e}"

def refresh_environment():
    """
    Reloads environment variables from the .env file
    """
    try:
        load_dotenv(override=True)
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            return "Environment refreshed successfully! API key is loaded."
        else:
            return "Environment refreshed, but no API key found."
    except Exception as e:
        return f"Error refreshing environment: {e}"

# Custom CSS for the app with purple theme
custom_css = """
:root {
    --primary-color: #8A2BE2;
    --secondary-color: #9370DB;
    --accent-color: #E6E6FA;
    --dark-purple: #4B0082;
    --light-purple: #D8BFD8;
    --text-color: #E6E6FA;
    --background-color: #1A1A1A;
    --card-bg: #2A2A2A;
    --input-bg: #333333;
    --shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    --glow: 0 0 15px rgba(138, 43, 226, 0.4);
}

body {
    background: linear-gradient(135deg, var(--background-color), #000000);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: var(--text-color);
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px;
}

/* Header Styling */
.header-container {
    display: flex;
    align-items: center;
    gap: 20px;
    background: linear-gradient(135deg, rgba(138, 43, 226, 0.2), rgba(75, 0, 130, 0.2));
    padding: 25px;
    border-radius: 20px;
    margin: 20px 20px 30px;
    box-shadow: var(--shadow);
    color: white;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(138, 43, 226, 0.2);
    backdrop-filter: blur(10px);
}

.header-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(45deg, transparent, rgba(138, 43, 226, 0.1), transparent);
    transform: translateX(-100%);
    transition: transform 0.6s;
}

.header-container:hover::before {
    transform: translateX(100%);
}

.header-container img {
    width: 60px;
    height: 60px;
    filter: drop-shadow(0 0 8px rgba(138, 43, 226, 0.5));
    transition: transform 0.3s ease;
}

.header-container:hover img {
    transform: scale(1.1) rotate(5deg);
}

.header-container h1 {
    margin: 0;
    font-size: 2.5rem;
    background: linear-gradient(to right, #FFF, var(--accent-color));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    text-shadow: 0 2px 4px rgba(0,0,0,0.5);
    letter-spacing: 0.5px;
}

.header-container p {
    margin: 8px 0 0;
    font-size: 1.1rem;
    color: var(--accent-color);
    letter-spacing: 0.3px;
    opacity: 0.9;
}

/* Configuration Accordion */
.config-accordion {
    border: none !important;
    border-radius: 15px !important;
    background: rgba(42, 42, 42, 0.7) !important;
    margin: 0 20px 30px !important;
    box-shadow: var(--shadow) !important;
    backdrop-filter: blur(10px) !important;
    transition: transform 0.3s ease, box-shadow 0.3s ease !important;
    border: 1px solid rgba(138, 43, 226, 0.2) !important;
}

.config-accordion:hover {
    transform: translateY(-2px) !important;
    box-shadow: var(--glow) !important;
}

.config-accordion > .label-wrap {
    background: linear-gradient(135deg, rgba(138, 43, 226, 0.3), rgba(75, 0, 130, 0.3)) !important;
    color: white !important;
    border-radius: 12px !important;
    font-weight: bold !important;
    padding: 15px 20px !important;
}

.config-accordion .prose {
    color: var(--text-color) !important;
    padding: 20px !important;
}

/* Main Content Layout */
.main-content {
    gap: 30px !important;
    margin: 20px !important;
    padding: 10px !important;
}

/* Input Column */
.input-column, .output-column {
    background: var(--card-bg) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 20px !important;
    padding: 25px !important;
    box-shadow: var(--shadow) !important;
    transition: transform 0.3s ease, box-shadow 0.3s ease !important;
    border: 1px solid rgba(138, 43, 226, 0.2) !important;
}

.input-column:hover, .output-column:hover {
    transform: translateY(-3px) !important;
    box-shadow: var(--glow) !important;
}

.upload-box {
    border: 2px dashed rgba(138, 43, 226, 0.4) !important;
    border-radius: 15px !important;
    transition: all 0.3s ease !important;
    padding: 20px !important;
    background: var(--input-bg) !important;
}

.upload-box:hover {
    border-color: var(--primary-color) !important;
    box-shadow: var(--glow) !important;
    background: rgba(51, 51, 51, 0.8) !important;
}

.prompt-input textarea {
    border: 2px solid rgba(138, 43, 226, 0.3) !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    background-color: var(--input-bg) !important;
    color: var(--text-color) !important;
    padding: 15px !important;
    font-size: 1.1rem !important;
}

.prompt-input textarea:focus {
    border-color: var(--primary-color) !important;
    box-shadow: var(--glow) !important;
    outline: none !important;
    background-color: rgba(51, 51, 51, 0.9) !important;
}

/* Generate Button Styling */
.generate-btn {
    background: linear-gradient(45deg, var(--primary-color), var(--secondary-color)) !important;
    color: white !important;
    border: none !important;
    border-radius: 30px !important;
    padding: 15px 30px !important;
    font-weight: bold !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    margin-top: 20px !important;
    box-shadow: 0 4px 15px rgba(138, 43, 226, 0.3) !important;
    position: relative !important;
    overflow: hidden !important;
}

.generate-btn::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    width: 100% !important;
    height: 100% !important;
    background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.2), transparent) !important;
    transform: translateX(-100%) !important;
    transition: transform 0.6s !important;
}

.generate-btn:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 6px 20px rgba(138, 43, 226, 0.4) !important;
    background: linear-gradient(45deg, var(--dark-purple), var(--primary-color)) !important;
}

.generate-btn:hover::before {
    transform: translateX(100%) !important;
}

.generate-btn:active {
    transform: translateY(1px) scale(0.98) !important;
}

/* Output Gallery */
.output-gallery {
    border-radius: 15px !important;
    overflow: hidden !important;
    background: var(--input-bg) !important;
    padding: 10px !important;
    transition: all 0.3s ease !important;
}

.output-gallery:hover {
    background: rgba(51, 51, 51, 0.8) !important;
}

/* Output Text */
.output-text textarea {
    border: 2px solid rgba(138, 43, 226, 0.3) !important;
    border-radius: 12px !important;
    background-color: var(--input-bg) !important;
    color: var(--text-color) !important;
    padding: 15px !important;
}

/* Labels and Text Elements */
label span {
    color: var(--accent-color) !important;
    font-weight: bold !important;
    font-size: 1.2rem !important;
    margin-bottom: 8px !important;
    display: inline-block !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3) !important;
    opacity: 0.9 !important;
}

/* Responsive design */
@media (max-width: 768px) {
    .header-container {
        flex-direction: column;
        text-align: center;
        padding: 20px;
    }
    
    .header-container h1 {
        font-size: 2rem;
    }
    
    .main-content {
        flex-direction: column !important;
        gap: 20px !important;
    }
    
    .input-column, .output-column {
        padding: 15px !important;
    }
    
    .generate-btn {
        width: 100% !important;
        padding: 12px 20px !important;
    }
}
"""

# Build a Blocks-based interface with custom embedded CSS
with gr.Blocks(css=custom_css) as demo:
    # Custom HTML header with proper class for styling
    gr.HTML(
    """
    <div class="header-container">
      <div>
          <img src="https://www.gstatic.com/lamda/images/gemini_favicon_f069958c85030456e93de685481c559f160ea06b.png" alt="Gemini logo">
      </div>
      <div>
          <h1>Gemini AI Image Studio</h1>
          <p>Transform your images with magical AI ✨</p>
      </div>
    </div>
    """
    )
    
    with gr.Accordion("⚠️ API Configuration ⚠️", open=True, elem_classes="config-accordion"):
        api_key_input = gr.Textbox(
            label="Gemini API Key",
            placeholder="Enter your Gemini API key here",
            type="password"
        )
        with gr.Row():
            save_api_key_button = gr.Button("Save API Key")
            refresh_button = gr.Button("🔄 Refresh Environment")
        api_key_output = gr.Textbox(label="Status", interactive=False)

        # Save API key and then refresh environment
        def save_and_refresh(api_key):
            save_result = update_env_file(api_key)
            refresh_result = refresh_environment()
            return f"{save_result}\n{refresh_result}"

        save_api_key_button.click(
            fn=save_and_refresh,
            inputs=[api_key_input],
            outputs=[api_key_output]
        )
        
        refresh_button.click(
            fn=refresh_environment,
            inputs=[],
            outputs=[api_key_output]
        )
        
        gr.Markdown("""
    - To use this app, you need to add your Gemini API key.
    - Get your API key from [Google AI Studio](https://aistudio.google.com/apikey)
    """)

    with gr.Row(elem_classes="main-content"):
        with gr.Column(elem_classes="input-column"):
            image_input = gr.Image(
                type="pil",
                label="Upload Image",
                image_mode="RGBA",
                elem_id="image-input",
                elem_classes="upload-box"
            )
            prompt_input = gr.Textbox(
                lines=2,
                placeholder="Describe how you want to transform your image...",
                label="Magic Prompt",
                elem_classes="prompt-input"
            )
            submit_btn = gr.Button("✨ Transform", elem_classes="generate-btn")
        
        with gr.Column(elem_classes="output-column"):
            output_gallery = gr.Gallery(label="Transformed Image", elem_classes="output-gallery")
            output_text = gr.Textbox(
                label="Gemini Thoughts", 
                placeholder="Text response will appear here if no image is generated.",
                elem_classes="output-text"
            )
            download_btn = gr.Button("📥 Download Image", elem_classes="generate-btn", visible=False)
            download_image = gr.Image(visible=False, type="pil", label="Download Image")
            
    # Set up the interaction with three outputs.
    submit_result = submit_btn.click(
        fn=process_image_and_prompt,
        inputs=[image_input, prompt_input],
        outputs=[output_gallery, output_text, download_image],
    )
    
    # Make download button visible when there's an image result
    submit_result.then(
        lambda img: gr.update(visible=img is not None),
        inputs=[download_image],
        outputs=[download_btn]
    )
    
    # Handle download button click
    download_btn.click(
        lambda x: x,
        inputs=[download_image],
        outputs=None,
        js="""
        async (img) => {
            if (!img) return;
            try {
                const response = await fetch(img.url || img);  // Use img.url if available
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'transformed_image.jpg';
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
            } catch (error) {
                console.error('Error downloading image:', error);
            }
        }
        """
    )

# Launch the app with a custom title
demo.queue(max_size=50).launch(share=False) 