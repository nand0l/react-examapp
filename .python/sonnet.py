import streamlit as st
import boto3
import json
import os
import re

# Initialize Boto3 session and DynamoDB resource
session = boto3.Session(profile_name='nlu', region_name='eu-west-1')
dynamodb = session.resource('dynamodb')
table = dynamodb.Table('ExamQuestionsV2')

# Define the file path for the prompt
prompt1_file_path = 'C:/Users/wnl/OneDrive/Documents/__SkillSoft/__AWS/MachineLearning/MLS-C01/.python/MLS-C01-a-prompt.md'
prompt1_file_path=prompt1_file_path.replace('\\', '\\\\')

# Read the input file
with open(prompt1_file_path, 'r') as file1:
    prompt1 = file1.read() + '\n'

# Set the output directory and file path
output_dir = '.\\output'
os.makedirs(output_dir, exist_ok=True)

def clean_json_string(json_string):
    # Replaces newlines and unescaped quotes within the JSON string values
    replacements = {"\n": "\\n", "\"": "\\\""}
    
    # To avoid escaping the outer quotes and object keys, replace only inside the values
    def replace_value(match):
        value = match.group(0)
        for old, new in replacements.items():
            value = value.replace(old, new)
        return value

    pattern = re.compile(r'(?<=:\s")([^"]+)(?=")')
    clean_string = pattern.sub(replace_value, json_string)
    return clean_string

def split_possible_answers(possible_answers):
    # Split the possible answers based on the pattern "A. ", "B. ", "C. ", etc.
    pattern = re.compile(r'(?=[A-Z]\.\s)')
    answers_list = pattern.split(possible_answers)[1:]
    return answers_list

def reformat_json(input_json, ExamID):
    data = json.loads(input_json)

    if isinstance(data['PossibleAnswers'], str):
        data['PossibleAnswers'] = split_possible_answers(data['PossibleAnswers'])

    question_id = data['QuestionID'].zfill(3)
    data['QuestionID'] = question_id
    data['ExamID'] = ExamID
    
    return data

def save_to_file_and_dynamodb(data, exam_id, question_id):
    output_file = os.path.join(output_dir, f"{exam_id}-{question_id}.json")
    
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
    #model_id= 'anthropic.claude-3-haiku-20240307-v1:0'

    input_data_json = json.dumps(input_data)

    response = client.invoke_model(contentType='application/json', body=input_data_json, modelId=model_id)

    inference_result = response['body'].read().decode('utf-8')
    content_text = json.loads(inference_result)['content'][0]['text']

    return content_text

def main():
    st.title('Invoke Bedrock Model')
    ExamID = st.text_input("ExamID", "MLS-C01-V2")
    prompt2 = st.text_area("Enter the question: ", "Paste the question", height=70)
    prompt = prompt1 + prompt2

    if st.button('Invoke Model'):
        content_text = invoke_bedrock_model(prompt)
        
        answer_output, output_file = process_input_json(ExamID, content_text)

        if answer_output:
            st.write(f"ExamID: {ExamID}")
            st.write(f"QuestionID: {answer_output['QuestionID']}")
            st.write(f"QuestionText: {answer_output['QuestionText']}")
            for idx, answer in enumerate(answer_output['PossibleAnswers']):
                st.write(f"PossibleAnswer {chr(65 + idx)}: {answer}")            
            st.write(f"Answer: {answer_output['Answer']}")
            st.write(f"CorrectAnswers: {answer_output['CorrectAnswers']}")
            st.write(f"Explanation: {answer_output['Explanation']}")

            st.write(f"Output written to: {output_file}")
        else:
            st.write("There was an error processing the JSON response.")

if __name__ == '__main__':
    main()
