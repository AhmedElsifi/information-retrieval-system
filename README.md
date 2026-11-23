# Information Retrieval System

A Python-based Information Retrieval (IR) system developed as an academic project. The system aims to retrieve and rank relevant documents based on user queries using classical Information Retrieval techniques.

## Overview

This project implements the core components of an Information Retrieval system, starting from document processing and indexing to document ranking and retrieval.

It also provides an interactive graphical interface built with **Streamlit**, allowing users to submit queries and explore retrieved documents.

## Features

The system is designed to include:

- **Document Loading:** Load and process the document collection.
- **Positional Indexing:** Build a positional inverted index to store term occurrences and their positions within documents.
- **Term Frequency (TF):** Calculate term frequencies across documents.
- **Document Statistics:** Calculate document frequency (DF) and inverse document frequency (IDF).
- **TF-IDF Weighting:** Calculate TF-IDF weights for terms in documents.
- **Document Normalization:** Normalize document vectors.
- **Query Processing:** Process and prepare user queries for retrieval.
- **Similarity Calculation:** Measure query-document similarity using cosine similarity.
- **Document Ranking:** Rank retrieved documents according to their relevance scores.
- **Graphical User Interface:** Provide an interactive search experience using Streamlit.

## Project Structure

```text
IR-Project/
├── dataset/
├── src/
│   └── ir_system/
│       ├── indexing/
│       ├── weighting/
│       ├── retrieval/
│       └── application_controller.py
├── gui/
│   ├── app.py
│   └── components/
├── output/
├── tests/
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Technologies

- **Python** – Core implementation.
- **NumPy** – Numerical computations.
- **Pandas** – Data processing and matrix representation.
- **Streamlit** – Graphical user interface.
- **Pytest** – Unit testing.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/USERNAME/information-retrieval-system.git
cd information-retrieval-system
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

Launch the Streamlit interface from the project root:

```bash
streamlit run gui/app.py
```

The application will open in your default browser.

## Testing

Run the test suite using:

```bash
pytest
```

## Project Scope

This project focuses on implementing and understanding fundamental Information Retrieval concepts, including indexing, term weighting, vector space representation, similarity measurement, and ranked retrieval.

The implementation is intended to demonstrate how these components work together to form a basic document retrieval system.

## Contributors

Developed as part of the Information Retrieval course project.

## License

This project is developed for academic and educational purposes.