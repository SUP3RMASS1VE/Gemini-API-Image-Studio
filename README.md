# 🎨 Gemini AI Image Studio

A sleek, modern web application that harnesses the power of Google's Gemini AI to transform and generate images based on your prompts. Built with Python and Gradio, featuring a beautiful dark-themed UI.
![Screenshot 2025-03-23 172811](https://github.com/user-attachments/assets/38277baf-4b7d-4ee5-a5e9-f2c9066e5312)

## ✨ Features

- **AI-Powered Image Transformation**: Transform your images using natural language prompts
- **Modern Dark UI**: Easy on the eyes with a professional, dark-themed interface
- **Real-time Processing**: Watch your transformations happen with a progress indicator
- **Easy Download**: Download your transformed images with one click
- **Responsive Design**: Works seamlessly on both desktop and mobile devices

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/SUP3RMASS1VE/Gemini-API-Image-Studio
cd gemini-image-studio
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running the App

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:7860
```

## 🎯 How to Use

1. **Configure API Key**:
   - Click on the "API Configuration" section
   - Enter your Gemini API key
   - Click "Save API Key"

2. **Transform Images**:
   - Upload an image using the upload box
   - Enter your transformation prompt in the "Magic Prompt" field
   - Click "✨ Transform" and watch the magic happen!

3. **Download Results**:
   - Once the transformation is complete, click "📥 Download Image"
   - Your transformed image will be saved to your downloads folder

## 🛠️ Technical Details

- **Frontend**: Built with Gradio for a seamless user interface
- **Backend**: Powered by Python and Google's Gemini AI
- **Image Processing**: Uses PIL (Python Imaging Library) for image handling
- **Security**: Environment variables for secure API key storage
- **Temporary File Management**: Automatic cleanup of temporary files

## 🎨 UI Features

- Dark theme with purple accents
- Glass-morphism effects
- Smooth animations and transitions
- Responsive layout
- Interactive elements with hover effects
- Progress indicators for transformations

## ⚠️ Important Notes

- The app requires a valid Gemini API key to function
- Image transformations may take a few moments depending on complexity
- Supported image formats: JPEG, PNG
- Maximum file size may be limited by the Gemini API

## 🔒 Security

- API keys are stored locally in the `.env` file
- No data is permanently stored on servers
- Temporary files are automatically cleaned up

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini AI for the image transformation capabilities
- Gradio for the web interface framework
- The open-source community for various dependencies

---

Made with ✨ and Python 
