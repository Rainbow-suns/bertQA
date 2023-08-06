import json

def count_samples(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)['data']
    total_samples = 0
    has_answer_samples = 0
    for article in data:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total_samples += 1
                if qa['answers']:
                    has_answer_samples += 1
    no_answer_samples = total_samples - has_answer_samples
    print(f'Total samples: {total_samples}')
    print(f'Samples with answers: {has_answer_samples}')
    print(f'Samples without answers: {no_answer_samples}')

if __name__ == '__main__':
    data_file = './dataset/train.json'
    count_samples(data_file)
