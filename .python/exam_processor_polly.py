import boto3
import json
import os
import time


def synthesize_text_to_s3_task(data, bucket_name):
    engine = 'standard'  # Valid Values: standard | neural | long-form | generative
    exam_id = data["ExamID"]
    question_id = data["QuestionID"]
    question_text = data["QuestionText"]
    explanation_text = data["Explanation"]
    question_output_file_name = f"{exam_id}-{question_id}-Q"
    explanation_text_output_file_name = f"{exam_id}-{question_id}-E"
    question_output_s3_key = f"polly-output/{
        exam_id}/{question_output_file_name}"
    explanation_output_s3_key = f"polly-output/{
        exam_id}/{explanation_text_output_file_name}"
    output_format = 'mp3'

    if engine == 'neural':
        voiceId = 'Joanna'  # Valid Values: Alexa | Joanna | Nicole | Russell |
    elif engine == 'generative':
        voiceId = 'Ruth'
    else:
        voiceId = 'Joey'

    polly_client = boto3.client('polly', region_name='us-east-1')

    question_response = polly_client.start_speech_synthesis_task(
        Engine=engine,
        Text=question_text,
        OutputFormat=output_format,
        VoiceId=voiceId,
        OutputS3BucketName=bucket_name,
        OutputS3KeyPrefix=question_output_s3_key
    )

    explanation_response = polly_client.start_speech_synthesis_task(
        Engine=engine,
        Text=explanation_text,
        OutputFormat=output_format,
        VoiceId=voiceId,
        OutputS3BucketName=bucket_name,
        OutputS3KeyPrefix=explanation_output_s3_key
    )

    question_url = f"https://{bucket_name}/{question_output_s3_key}.{
        question_response['SynthesisTask']['TaskId']}.mp3"
    explanation_url = f"https://{bucket_name}/{explanation_output_s3_key}.{
        explanation_response['SynthesisTask']['TaskId']}.mp3"

    print(f"Bucketname: {bucket_name}")
    print(f"Exam-ID: {exam_id}")
    print(f"Question-ID: {question_id}")
    print(f"Question-URL: {question_url}")
    print(f"Explanation-URL: {explanation_url}")

    update_dynamodb_item(exam_id, question_id, question_url, explanation_url)


def update_dynamodb_item(exam_id, question_id, question_url, explanation_url):
    dynamodb = boto3.resource('dynamodb', region_name='eu-west-1')
    table = dynamodb.Table('ExamQuestionsV2')

    response = table.update_item(
        Key={
            'ExamID': exam_id,
            'QuestionID': question_id
        },
        UpdateExpression="set #question_url = :q_url, #explanation_url = :e_url",
        ExpressionAttributeNames={
            '#question_url': 'QuestionURL',
            '#explanation_url': 'ExplanationURL'
        },
        ExpressionAttributeValues={
            ':q_url': question_url,
            ':e_url': explanation_url
        },
        ReturnValues="UPDATED_NEW"
    )


def process_folder_and_synthesize(folder_path, bucket_name):
    if not os.path.exists(folder_path):
        print(f"Error: The specified directory does not exist: {folder_path}")
        return

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    print(f"Processing file: {filename}")
                    synthesize_text_to_s3_task(data, bucket_name)
                    time.sleep(5)
            except json.JSONDecodeError:
                print(f"Skipping file due to JSON decode error: {filename}")
            except Exception as e:
                print(f"Error processing file {filename}: {e}")


def move_processed_files(folder_path):
    processed_folder = os.path.join(folder_path, 'processed')
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            destination_path = os.path.join(processed_folder, filename)
            try:
                os.rename(file_path, destination_path)
                print(f"Moved {file_path} to {destination_path}")
            except Exception as e:
                print(f"Error moving file {filename}: {e}")


def process_polly(ExamID, bucket_name):
    # Getting the script directory and one level up to set the base path
    script_path = os.path.abspath(__file__)
    base_path = os.path.abspath(os.path.join(
        script_path, os.pardir, os.pardir))
    folder_path = os.path.join(base_path, ExamID, 'output')

    process_folder_and_synthesize(folder_path, bucket_name)
    move_processed_files(folder_path)
