def add_line_break(sentence, max_words_per_line):
    # Split the sentence into words
    words = sentence.split()

    # Initialize an empty result string and a word count
    result = ""
    word_count = 0

    # Iterate through the words and add a line break when reaching the maximum words per line
    for word in words:
        result += word + " "
        word_count += 1
        if word_count >= max_words_per_line:
            result += "\n"
            word_count = 0

    return result.strip()