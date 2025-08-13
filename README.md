# SECURE-SURF 🛡️

A comprehensive phishing detection system that combines machine learning, web applications, and browser extensions to identify malicious websites in real-time.


## 🚀 Features

- **Real-time URL Analysis**: Advanced machine learning models to detect phishing websites
- **Multi-platform Support**: Web interface and Chrome extension
- **Comprehensive Feature Extraction**: 74+ features including URL characteristics, HTML content, and domain properties
- **Modern UI**: Interactive 3D animated interface using Three.js
- **REST API**: Easy integration with other systems
- **Browser Extension**: Automatic protection while browsing

## 🏗️ Architecture

### Core Components

1. **Flask Web Application** - Main server with REST API
2. **Machine Learning Models** - CNN, MLP, and Random Forest classifiers
3. **Feature Extraction Engine** - 74 different feature analyzers
4. **Chrome Extension** - Browser integration for real-time protection
5. **Web Interface** - Modern responsive UI with 3D animations

### Technology Stack

- **Backend**: Python, Flask, scikit-learn, PyTorch
- **Frontend**: HTML5, CSS3, JavaScript, Three.js
- **ML Libraries**: pandas, numpy, BeautifulSoup
- **Network Analysis**: socket, ssl, dns.resolver, whois
- **Browser Extension**: Chrome Extension API (Manifest V3)

## 📋 Prerequisites

- Python 3.8+
- Node.js (for extension development)
- Chrome Browser (for extension)
- Internet connection (for real-time analysis)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/secure-surf.git
cd secure-surf
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download/Train Models

The project uses pre-trained models. You have two options:

**Option A: Use existing models (if available)**
- Ensure `best_rf_model.joblib` is in the project root

**Option B: Train new models**
```bash
python Model_test.py
```

### 4. Prepare Dataset (Optional)

If training new models, you'll need:
- `original_new_phish_25k.csv` - Phishing dataset
- `legit_data.csv` - Legitimate websites dataset

## 🚀 Usage

### Web Application

1. **Start the Flask server:**
```bash
python app.py
```

2. **Access the web interface:**
   - Open your browser and go to `http://localhost:2002`
   - Enter a URL in the input field
   - Click "Submit" to analyze

3. **API Usage:**
```bash
curl -X POST http://localhost:2002/check_phishing \
  -H "Content-Type: application/json" \
  -d '{"url": "example.com"}'
```

### Chrome Extension

1. **Load the extension:**
   - Open Chrome and go to `chrome://extensions/`
   - Enable "Developer mode"
   - Click "Load unpacked" and select the `Extension` folder

2. **Use the extension:**
   - The extension automatically analyzes the current page
   - Click the extension icon for manual checks
   - Alerts will show if a site is detected as phishing

## 📁 Project Structure

```
secure-surf/
├── app.py                          # Main Flask application
├── run.py                          # URL processing pipeline
├── exten.py                        # Extension-compatible processing
├── Model_test.py                   # ML model training and evaluation
├── Feature_extraction_ff1.py       # Feature extraction orchestrator
├── feature_init_ff1.py             # 74 feature extraction functions
├── result.py                       # Domain analysis utilities
├── best_rf_model.joblib            # Trained Random Forest model
├── best_initial_model.joblib       # Alternative trained model
├── templates/                      # HTML templates
│   ├── home.html                   # Main interface
│   ├── about.html                  # About page
│   ├── contact.html                # Contact form
│   ├── result.html                 # Results display
│   └── dev.html                    # Development template
├── static/                         # Static assets
│   ├── *.css                       # Stylesheets
│   ├── main.js                     # JavaScript utilities
│   └── Logo2.png                   # Project logo
├── Extension/                      # Chrome extension
│   ├── manifest.json               # Extension configuration
│   ├── background.js               # Service worker
│   ├── content.js                  # Content script
│   ├── popup.html                  # Extension popup
│   ├── popup.js                    # Popup functionality
│   └── icons/                      # Extension icons
└── README.md                       # This file
```

## 🎯 Feature Extraction

The system analyzes 74 different features across multiple categories:

### URL Features
- URL length and special characters
- SSL certificate analysis
- Domain age and registration details
- Subdomain analysis

### HTML Content Features
- Form elements and input fields
- JavaScript and executable files
- Image and media elements
- HTML structure analysis

### Network Features
- DNS records (MX, TXT, NS)
- IP address analysis
- SSL certificate details
- SMTP server configuration

### Security Features
- Known malicious patterns
- URL shortening services
- Suspicious redirects
- Abnormal URL structures

## 🧠 Machine Learning Models

### Available Models

1. **Random Forest** (Production model)
   - High accuracy and fast inference
   - Robust to feature variations
   - Used for real-time detection

2. **Deep Learning Models** (Research/Comparison)
   - CNN (Convolutional Neural Network)
   - MLP (Multi-Layer Perceptron)
   - ResidualMLP, DropoutMLP, BatchNormMLP

### Model Performance

The system achieves high accuracy on the test dataset with comprehensive feature analysis. Model comparison and evaluation results are available in `Model_test.py`.

## 🔧 API Reference

### POST /check_phishing

Analyze a URL for phishing indicators.

**Request:**
```json
{
  "url": "https://example.com"
}
```

**Response:**
```json
{
  "result_text": "The URL is predicted as a legitimate domain🟢.",
  "additional_info": {
    "domain": "example.com",
    "ip": "93.184.216.34",
    "Domain Age": "28",
    "num_sub_domains": "2",
    "domain_reg_length": "365",
    "ip_counts": "1",
    "ssl_update_age(In Days)": "30",
    "num_smtp_servers": "0"
  }
}
```

## 🔒 Security Considerations

- The system analyzes URLs without storing personal data
- All network requests are made server-side for security
- Chrome extension requests permission only for active tabs
- No sensitive information is logged or stored

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Development

### Adding New Features

1. **New Feature Extractors**: Add functions to `feature_init_ff1.py`
2. **UI Components**: Modify templates in `templates/` directory
3. **API Endpoints**: Extend `app.py` with new routes
4. **Extension Features**: Update files in `Extension/` directory

### Testing

```bash
# Test individual components
python -c "import run; print(run.process_url_input('google.com'))"

# Test API endpoint
curl -X POST http://localhost:2002/check_phishing \
  -H "Content-Type: application/json" \
  -d '{"url": "google.com"}'
```

## 🐛 Troubleshooting

### Common Issues

1. **Model not found error**
   - Ensure `best_rf_model.joblib` exists in project root
   - Run `python Model_test.py` to train new models

2. **Extension not working**
   - Check if Flask server is running on port 2002
   - Verify extension permissions in Chrome

3. **Feature extraction errors**
   - Check internet connection for WHOIS/DNS queries
   - Some features may fail for inaccessible domains (expected behavior)

4. **Import errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Activate virtual environment before running

## 📊 Dataset Information

The system is trained on:
- **Phishing samples**: 25,000 malicious URLs
- **Legitimate samples**: Verified safe websites
- **Features**: 74 extracted features per URL

## 🔮 Future Enhancements

- [ ] Real-time model updates
- [ ] Firefox extension support
- [ ] Mobile app integration
- [ ] Advanced visualization dashboard
- [ ] Batch URL analysis API
- [ ] Integration with threat intelligence feeds

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

