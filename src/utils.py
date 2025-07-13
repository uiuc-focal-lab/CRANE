import json
import os
import re

def load_data_by_name(task, do_cot = True):
    from datasets import load_dataset, Dataset

    if task == 'gsm8k':
        data = []
        i = 0
        for row in load_dataset('gsm8k', 'main', split='test'):
            data.append(dict(question = row['question'], answer = row['answer'], idx = i))
            i += 1
        return Dataset.from_list(data)
    elif task == 'deepmind_math':
        data = []
        i = 0
        for row in load_dataset('deepmind/math_dataset', 'calculus__differentiate')['test']:
            data.append(dict(question = row['question'][2:-3], answer = row['answer'][2:-3], idx = i))
            i += 1
        return Dataset.from_list(data)

    elif task == 'spider':
        data = []
        i = 0
        for row in load_dataset("richardr1126/spider-context-validation", split="validation"):
            if not do_cot:
                prompt = f"### Database ID: {row['db_id']}\nSQL tables information: \n{row['db_info']}\n### Question: {row['question']}\nSQL:"
            else:
                prompt = f"### Database ID: {row['db_id']}\nSQL tables information: \n{row['db_info']}\n### Question: {row['question']}"

            data.append(dict(question = prompt, answer = row['ground_truth'], idx = i, db_info = row['db_info']))
            i += 1
        return Dataset.from_list(data)
    elif task == 'gsm_symbolic':
        data = []
        i = 0
        folder_path = 'gsm_symbolic/'  
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                with open(os.path.join(folder_path, filename), 'r') as f:
                    json_data = json.load(f)
                    question = json_data['question_parsed']
                    data.append(dict(question= question, answer=json_data['answer_parsed'], idx=i, variable_types = str(json_data['variable_types'])))
                    i += 1
        ds = Dataset.from_list(data)
        return ds
    elif task == 'fol':
        data = []
        i = 0 
        for problem in load_dataset("tasksource/folio", split="validation"):
            context = problem['premises']
            question = f"Based on the above information, is the following statement true, false, or uncertain? {problem['conclusion']}"
            prompt = 'Problem:\n[[PROBLEM]]\nQuestion:\n[[QUESTION]]\n###'
            prompt = prompt.replace('[[PROBLEM]]', context).replace('[[QUESTION]]', question.strip())
            data.append(dict(question = prompt, answer = problem['label'], idx = i))
            i += 1
        return Dataset.from_list(data)

    elif task == 'math':
        data = []
        for row in load_dataset("appier-ai-research/robust-finetuning", "math")['test']:
            row = dict(row)
            row['question'] = row['problem']
            row['answer'] = row['solution']
            data.append(row)
        return Dataset.from_list(data)
    elif task == 'ddxplus':
        data = []
        for row in load_dataset('appier-ai-research/StreamBench',
                            "ddxplus",
                            split='test'
                        ):
            row = dict(row)
            row['question'] = row['PATIENT_PROFILE']
            row['answer'] = row['PATHOLOGY']
            data.append(row)
        return Dataset.from_list(data)
    elif task == 'lastletter':
        return load_dataset('ChilleD/LastLetterConcat', split='test')
    elif task == 'multifin':
        data = []
        for row in load_dataset('ChanceFocus/flare-multifin-en',
                            split='test'
                        ):
            row = dict(row)
            row['question'] = row['text']
            row['answer'] = row['answer'].replace('&', 'and')
            data.append(row)
        return Dataset.from_list(data)
    elif task == 'multiarith':
        data = []
        for row in load_dataset('ChilleD/MultiArith', split='test'):
            row = dict(row)
            row['question'] = row['question']
            row['answer'] = row['final_ans']
            data.append(row)
        return Dataset.from_list(data)
    elif task == 'shuffleobj':
        data = []
        choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        for row in load_dataset('tasksource/bigbench', 'tracking_shuffled_objects', split='validation'):
            row = dict(row)
            answer_choice = '\n'.join([ '{}) {}'.format(l, t) for l, t in zip(choices, row['multiple_choice_targets'])])
            row['question'] = row['inputs']+'\n'+answer_choice
            row['answer'] = choices[np.argmax(row['multiple_choice_scores'])]
            data.append(row)
        return Dataset.from_list(data)
    elif task == 'dateunder':
        data = []
        choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        for row in load_dataset('tasksource/bigbench', 'date_understanding', split='validation'):
            row = dict(row)
            answer_choice = '\n'.join([ '{}) {}'.format(l, t) for l, t in zip(choices, row['multiple_choice_targets'])])
            row['question'] = row['inputs']+'\n'+answer_choice
            row['answer'] = choices[np.argmax(row['multiple_choice_scores'])]
            data.append(row)
        return Dataset.from_list(data)
    elif task == 'csqa':
        data = []
        for row in load_dataset('tau/commonsense_qa', split='validation'):
            row = dict(row)
            answer_choice = '\n'.join([ '{}) {}'.format(l, t) for l, t in zip(row['choices']['label'], row['choices']['text'])])
            row['question'] = row['question']+'\n'+answer_choice
            row['answer'] = row['answerKey']
            data.append(row)
        return Dataset.from_list(data)
    elif task == 'sports':
        data = []
        for row in load_dataset('tasksource/bigbench',
                                'sports_understanding',
                            split='validation',
                            trust_remote_code=True
                        ):
            row = dict(row)
            row['question'] = row['inputs'].replace('Determine whether the following statement or statements are plausible or implausible:','').replace('Statement: ','').replace('Plausible/implausible?','').strip()
            row['answer'] = 'yes' if row['targets'][0] == 'plausible' else 'no'
            data.append(row)
        return Dataset.from_list(data)
    elif task == 'task280':
        with open('data/task280_stereoset_classification_stereotype_type.json', 'r') as f:
            raw_data = json.load(f)
        data = []
        for row in raw_data['Instances'][:1000]:
            data.append({
                'question': row['input'],
                'answer': row['output'][0].lower()
            })
        return Dataset.from_list(data)
    elif task == 'conll2003':
        data = []
        for row in load_dataset("eriktks/conll2003", split="test"):
            row = dict(row)
            question = ' '.join(row['tokens'])
            row['question'] = question
            row['answer'] = row['ner_tags']
            data.append(row)
        return Dataset.from_list(data)
    elif task == 'api-bank':
        data = []
        with open('API-Bank/test.jsonl', 'r') as f:
            for line in f:
                payload = json.loads(line)
                data.append(payload)
        return data

    raise ValueError("%s is not in supported list" % task)


