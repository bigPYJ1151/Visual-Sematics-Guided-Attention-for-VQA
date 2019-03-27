
import json
import itertools
from collections import Counter
from text_preprocess import prepare_answers, prepare_questions
import yaml
import os
from addict import Dict

CONFIG_VQA = Dict(yaml.load(open('config.yaml'))['VQA'])

def vocab_extract(iterable, top_k = None, start = 0):
    all_tokens = itertools.chain.from_iterable(iterable)
    counter = Counter(all_tokens)

    if top_k:
        most_common = counter.most_common(top_k)
        most_common = (t for t, c in most_common)
    else:
        most_common = counter.keys()
    
    tokens = sorted(most_common, key=lambda x: (counter[x], x), reverse=True)
    vocab = {t: i for i, t in enumerate(tokens, start=start)}
    return vocab

def main():
    questions = os.path.join(CONFIG_VQA.DATASET.VQA, "OpenEnded_mscoco_train2014_questions.json")
    answers = os.path.join(CONFIG_VQA.DATASET.VQA, "mscoco_train2014_annotations.json")

    with open(questions, 'r') as fd:
        questions = json.load(fd)
    with open(answers, 'r') as fd:
        answers = json.load(fd)

    questions = prepare_questions(questions)
    answers = prepare_answers(answers)

    question_vocab = vocab_extract(questions, start=1)
    answer_vocab = vocab_extract(answers, top_k=CONFIG_VQA.DATASET.VOCAB_NUM)

    vocabs = {
        'question': question_vocab,
        'answer': answer_vocab,
    }
    with open(CONFIG_VQA.DATASET.VOCAB, 'w') as fd:
        json.dump(vocabs, fd)

if __name__ == '__main__':
    main()