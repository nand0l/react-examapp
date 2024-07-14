import re
import os

def process_exam_file(file_name):
    # Getting the script directory and one level up to set the base path
    script_path = os.path.abspath(__file__)
    base_path = os.path.abspath(os.path.join(script_path, os.pardir, os.pardir))

    file_path = os.path.join(base_path, file_name)
    prep_folder = 'prep'

    def split_questions_from_file(file_path):
        # Read the content of the file
        with open(file_path, 'r') as file:
            content = file.read()

        # Split content by 'Question:' followed by a number and newline
        questions = re.split(r'(\n## Question:\d+\n)', content)

        # Initialize a list to store formatted questions and their numbers
        formatted_questions = []

        # Loop through the splits, extracting the question number and question text
        for i in range(1, len(questions), 2):
            question_header = questions[i]
            question_body = questions[i + 1].strip()

            # Extract the question number
            match = re.search(r'## Question:(\d+)', question_header)
            if match:
                QuestionID = (match.group(1)).zfill(3)
                formatted_question = f"Question:{QuestionID}\n{question_body}"
                formatted_questions.append((QuestionID, formatted_question))

        return formatted_questions

    def read_first_line(file_path):
        with open(file_path, 'r') as file:
            first_line = file.readline().strip()
        return first_line

    ExamID = read_first_line(file_path)[2:]
    full_path = os.path.join(base_path, ExamID, prep_folder)

    # Create the output folder if it doesn't exist
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    # Split the file into the different questions
    questions = split_questions_from_file(file_path)

    # Display the split questions
    for QuestionID, question in questions:
        # Specify the file name
        file_name = ExamID+'-'+QuestionID+'.md'

        # Combine the full path with the file name
        file_path = os.path.join(full_path, file_name)

        print(f"ExamID: {ExamID}")
        print(f"QuestionID: {QuestionID}")
        try:
            with open(file_path, 'w') as file:
                file.write("## "+question)
            print(f"File created successfully: {file_path}")
        except Exception as e:
            print(f"An error occurred while creating the file: {e}")
        print("\n" + "="*20 + "\n")

    return ExamID
