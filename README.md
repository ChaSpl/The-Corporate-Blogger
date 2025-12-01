# The Corporate Blogger

AI agents weave your brand's evolving narrative journey into powerful storytelling. This is a powerful agent architecture that generates professional corporate blog posts using Google's AI API. Perfect for businesses, marketers, and content creators who need consistent, high-quality blog content.

## ✨ Features

- **AI-Powered Content Generation** - Creates professional blog posts using Google's AI models
- **Corporate Tone** - Maintains professional, business-appropriate language
- **Customizable** - Easy to modify prompts and parameters
- **Secure** - API keys stored safely in environment variables
- **Simple Interface** - User-friendly command-line interface

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- Google AI Studio API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ChaSpl/The-Corporate-Blogger.git
   cd The-Corporate-Blogger

2. **Install required packages**
   ```bash 
   pip install google-generativeai python-dotenv

3. **Set up your environment variables**
    Create a .env file in the project root directory

    Add your Google API key:
    GOOGLE_API_KEY=your_actual_google_api_key

4. **Usage**
    Run the script:
    python BLOG_WRITER.py

    The program will guide you through:

    - Entering your blog topic
    - Specifying the blog length

## 🔒 Security

    Important: Never commit your .env file or share your API keys publicly.
    The .gitignore file is configured to exclude .env
    API keys are loaded from environment variables only
    Your secrets remain on your local machine

## 📝✨ Happy Blogging! 

