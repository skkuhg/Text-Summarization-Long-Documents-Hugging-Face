# Text Summarization for Long Documents

A comprehensive implementation of text summarization for long documents using Hugging Face transformers and a sliding window approach to handle token limitations.

## Overview

Traditional summarization models like BART or T5 have token limitations that prevent them from processing long documents directly. This project implements a sliding window approach to break long texts into manageable chunks, summarize each chunk, and combine the results into a coherent final summary.

## Features

- **Sliding Window Approach**: Handles documents longer than model token limits
- **Overlapping Chunks**: Ensures context preservation between chunks
- **Batch Processing**: Support for multiple documents
- **Customizable Parameters**: Adjustable chunk size, overlap, and summary length
- **GPU Acceleration**: Optional GPU support for faster processing
- **Export Functionality**: Save summaries to JSON format

## Installation

1. Clone this repository:
```bash
git clone https://github.com/skkuhg/Text-Summarization-Long-Documents-Hugging-Face.git
cd text-summarization-long-documents
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Open the Jupyter notebook `text_summarization_notebook.ipynb` and run the cells sequentially. The notebook includes:

1. **Setup and Installation**: Install required libraries
2. **Model Loading**: Load pre-trained BART model
3. **Sliding Window Implementation**: Core summarization function
4. **Example Usage**: Test with sample long document
5. **Advanced Features**: Batch processing and customization options

### Key Functions

#### `sliding_window_summarize(text, max_input_length=1024, max_output_length=200, stride=512)`

Main function for summarizing long documents.

**Parameters:**
- `text`: Input long text to be summarized
- `max_input_length`: Maximum tokens per chunk (default: 1024)
- `max_output_length`: Maximum tokens in output summary (default: 200)
- `stride`: Overlap between chunks (default: 512)

#### `batch_summarize_documents(documents, **kwargs)`

Process multiple documents in batch.

**Parameters:**
- `documents`: Dictionary of {document_name: document_text}
- `**kwargs`: Additional parameters for sliding_window_summarize

### Example

```python
from transformers import BartForConditionalGeneration, BartTokenizer, pipeline

# Load model
model_name = 'facebook/bart-large-cnn'
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

# Summarize long document
long_text = "Your very long document text here..."
summary = sliding_window_summarize(long_text)
print(summary)
```

## Model Information

This implementation uses Facebook's BART model fine-tuned on CNN/Daily Mail dataset (`facebook/bart-large-cnn`). The model is specifically designed for abstractive summarization tasks.

## Requirements

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- NumPy
- Datasets

See `requirements.txt` for specific versions.

## Performance Considerations

- **GPU Acceleration**: Enable CUDA for faster processing if available
- **Memory Usage**: Large models require significant RAM/VRAM
- **Processing Time**: Long documents may take several minutes to process
- **Batch Processing**: Process multiple chunks simultaneously for better efficiency

## Customization

### Parameters Tuning

- **max_input_length**: Increase for models with larger context windows
- **stride**: Adjust overlap between chunks (smaller = more overlap)
- **max_output_length**: Control summary verbosity

### Alternative Models

The code can be adapted for other summarization models:
- T5 (`t5-base`, `t5-large`)
- Pegasus (`google/pegasus-cnn_dailymail`)
- Custom fine-tuned models

## File Structure

```
├── text_summarization_notebook.ipynb  # Main Jupyter notebook
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
└── .gitignore                        # Git ignore rules
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## Potential Improvements

- **Fine-tuning**: Train on domain-specific data
- **Preprocessing**: Advanced text cleaning and normalization
- **Evaluation Metrics**: Implement ROUGE scores for quality assessment
- **Alternative Architectures**: Experiment with newer models
- **API Integration**: Create REST API for summarization service

## Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [BART Paper](https://arxiv.org/abs/1910.13461)
- [Summarization Task Guide](https://huggingface.co/tasks/summarization)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Hugging Face for the transformers library
- Facebook AI for the BART model
- The open-source community for inspiration and resources
