import time
import boto3
import json
import os
import re

def process_exam_files(ExamID):
    # Getting the script directory and one level up to set the base path
    script_path = os.path.abspath(__file__)
    base_path = os.path.abspath(os.path.join(script_path, os.pardir, os.pardir))

    # Prompt file name and paths
    prompt1_filename = 'MLS-C01-a-prompt.md'
    prompt1_file_path = os.path.join(base_path, '.python', prompt1_filename)

    # Folders for preparation and output
    prep_folder = 'prep'
    output_folder = 'output'
    full_path = os.path.join(base_path, ExamID, prep_folder)
    output_path = os.path.join(base_path, ExamID, output_folder)
    os.makedirs(output_path, exist_ok=True)

    # Initialize Boto3 session and DynamoDB resource
    session = boto3.Session(profile_name='nlu', region_name='eu-west-1')
    dynamodb = session.resource('dynamodb')
    table = dynamodb.Table('ExamQuestionsV2')

    def read_file_content(file_path):
        with open(file_path, 'r') as file:
            return file.read()

    def clean_json_string(json_string):
        replacements = {"\n": "\\n", "\"": "\\\""}
        def replace_value(match):
            value = match.group(0)
            for old, new in replacements.items():
                value = value.replace(old, new)
            return value
        pattern = re.compile(r'(?<=:\s")([^"]+)(?=")')
        return pattern.sub(replace_value, json_string)

    def split_possible_answers(possible_answers):
        pattern = re.compile(r'(?=[A-Z]\.\s)')
        return pattern.split(possible_answers)[1:]

    def reformat_json(input_json, ExamID):
        data = json.loads(input_json)
        if isinstance(data['PossibleAnswers'], str):
            data['PossibleAnswers'] = split_possible_answers(data['PossibleAnswers'])
        data['QuestionID'] = data['QuestionID'].zfill(3)
        data['ExamID'] = ExamID
        return data

    def save_to_file_and_dynamodb(data, exam_id, question_id):
        output_file = os.path.join(output_path, f"{exam_id}-{question_id}.json")
        with open(output_file, 'w') as file:
            json.dump(data, file, indent=4)
        table.put_item(Item=data)
        return output_file

    def process_input_json(ExamID, input_json):
        cleaned_content = clean_json_string(input_json)
        try:
            answer_output = reformat_json(cleaned_content, ExamID)
            output_file = save_to_file_and_dynamodb(answer_output, ExamID, answer_output['QuestionID'])
            return answer_output, output_file
        except json.JSONDecodeError as e:
            print(f"JSON decoding failed: {e}")
            print("Check the format of your JSON string.")
            return None, None

    def invoke_bedrock_model(prompt):
        client = boto3.client('bedrock-runtime', region_name="us-east-1")
        input_data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt.replace("\n", "\\n")
                        }
                    ]
                }
            ],
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2000
        }
        model_id = 'anthropic.claude-3-5-sonnet-20240620-v1:0'
        input_data_json = json.dumps(input_data)
        response = client.invoke_model(contentType='application/json', body=input_data_json, modelId=model_id)
        inference_result = response['body'].read().decode('utf-8')
        content_text = json.loads(inference_result)['content'][0]['text']
        return content_text

    prompt1 = read_file_content(prompt1_file_path) + '\n'
    
    for filename in os.listdir(full_path):
        if filename.endswith('.md'):
            file_path = os.path.join(full_path, filename)
            prompt2 = read_file_content(file_path)
            prompt = prompt1 + prompt2
            content_text = invoke_bedrock_model(prompt)
            answer_output, output_file = process_input_json(ExamID, content_text)
            
            if answer_output:
                print(f"Processed file: {filename}")
            else:
                print(f"Error processing file: {filename}")
            
            time.sleep(5)
