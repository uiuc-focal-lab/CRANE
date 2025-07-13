import srsly
import fire


def main(first, second):
    first = srsly.read_jsonl(first)
    second = srsly.read_jsonl(second)

    first_only = []
    second_only = []
    both = []
    neither = []

    for i, (one, two) in enumerate(zip(first, second)):
        if one['correct'] == False and two['correct'] == True:
            second_only.append(i)
        
        elif one['correct'] == True and two['correct'] == False:
            first_only.append(i)
        
        elif one['correct'] and two['correct']:
            both.append(i)
        
        else:
            neither.append(i)

    print('Both (Total, ids) ', len(both), both )
    print()
    print('Neither (Total, ids) ', len(neither), neither)
    print()
    print('First Only (Total, ids) ', len(first_only), first_only)
    print()
    print('Second Only (Total, ids) ', len(second_only), second_only)

if __name__ == '__main__':
    fire.Fire(main)
