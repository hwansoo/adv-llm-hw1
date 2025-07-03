import json

with open('qa_list.json', 'r') as f:
    qa_list = json.load(f)

print(f'Total questions: {len(qa_list)}')

# Analyze answer patterns
answers = [qa['answer'] for qa in qa_list]
no_answer_count = sum(1 for ans in answers if ans == 'No Answer')
print(f'No Answer responses: {no_answer_count} ({no_answer_count/len(answers)*100:.1f}%)')

# Check answer lengths
answer_lengths = [len(ans.split()) for ans in answers if ans != 'No Answer']
print(f'Average answer length: {sum(answer_lengths)/len(answer_lengths):.1f} words')
print(f'Answer length range: {min(answer_lengths)}-{max(answer_lengths)} words')

# Sample some questions to understand patterns
print('\nSample questions and answers:')
for i in range(0, min(5, len(qa_list))):
    print(f'Q: {qa_list[i]["question"]}')
    print(f'A: {qa_list[i]["answer"]}')
    print()