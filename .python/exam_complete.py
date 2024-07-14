import exam_processor_reader
import exam_processor
import exam_processor_polly

def main():
    exam_file = 'MLS-C01-a-1to12.md'
    ExamID = exam_processor_reader.process_exam_file(exam_file)
    print(f"ExamID: {ExamID}")
    
    exam_processor.process_exam_files(ExamID)
    
    bucket_name = 'amazoninstructor.info'
    exam_processor_polly.process_polly(ExamID, bucket_name)

if __name__ == '__main__':
    main()
