import json


def generate_dictionary(intersection_vocab_file, output_dict_file):
    # Read intersection vocabulary
    vocabulary = []
    with open(intersection_vocab_file, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word:  # Skip empty lines
                vocabulary.append(word)

    # Create dictionary with word-to-index mapping
    # Start from index 1, reserve 0 for padding
    word_dict = {word: idx + 1 for idx, word in enumerate(vocabulary)}

    # Save dictionary to JSON file
    with open(output_dict_file, 'w', encoding='utf-8') as f:
        json.dump(word_dict, f, ensure_ascii=False, indent=2)

    return word_dict


# Usage example
word_dict = generate_dictionary('data/vocab/intersection_vocab.txt', 'data/dict/word_dict.json')
print(f"Dictionary size: {len(word_dict)}")