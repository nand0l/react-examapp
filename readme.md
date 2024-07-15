# AWS Exam Processing Automation

The ExamID is set from line 1 of the inputfile, 
The inputfile is set at line 7 of `exam_complete.py`

This repository contains a suite of Python scripts designed to automate the processing of exam data for AWS certification exams. The automation covers several steps including reading and processing exam questions, storing and managing data in AWS DynamoDB, and generating audio files using AWS Polly.

## Key Features

1. **Exam File Processing**:
   - Extracts and formats questions from markdown files.
   - Stores processed questions into structured JSON format.
   - Saves processed data to AWS DynamoDB for easy retrieval and management.

2. **Text-to-Speech Synthesis**:
   - Converts exam questions and explanations into audio files using AWS Polly.
   - Stores generated audio files in an S3 bucket.
   - Updates DynamoDB with URLs to the generated audio files for easy access.

3. **Automated Workflows**:
   - Organized modular scripts to handle different stages of the processing pipeline.
   - Scripts are designed to be easily invoked sequentially, ensuring smooth data processing and audio generation.

## Repository Structure

* **`exam_processor_reader.py`**:
    - Contains the function to read and process the exam file, extracting questions and formatting them for further processing.
    - Returns an `ExamID` for use in subsequent steps.

* **`exam_processor.py`**:
    - Processes exam files based on the given `ExamID`.
    - Reads prompts, invokes language models, and saves the results to DynamoDB.

* **`exam_processor_polly.py`**:
    - Handles the text-to-speech synthesis tasks.
    - Converts questions and explanations to audio files using AWS Polly.
    - Moves processed files to a designated folder and updates DynamoDB with audio file URLs.

* **`exam_complete.py`**:
    - Orchestrates the entire workflow by calling functions from the other scripts in sequence.
    - Ensures seamless processing from reading exam files to generating and storing audio files.

## How to Use

1. **Prerequisites**:
   - Ensure you have AWS credentials configured with appropriate permissions.
   - Set up a DynamoDB table named `ExamQuestionsV2` .
   - Configure an S3 bucket for storing Polly output.

2. **Running the Scripts**:
   - Execute `exam_complete.py` to run the complete processing workflow.
   - The script will prompt for an `ExamCode` , process the exam file, and generate audio files, storing all results in DynamoDB and S3.

## Requirements

* Python 3.6+
* Boto3
* AWS Account with necessary permissions for DynamoDB and Polly

## Setup

1. Clone the repository:

```sh
   git clone https://github.com/your-repo/aws-exam-processing-automation.git
   cd aws-exam-processing-automation
   ```

2. Install dependencies:

```sh
   pip install boto3
   ```

3. Configure AWS credentials:

```sh
   aws configure
   ```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License.
