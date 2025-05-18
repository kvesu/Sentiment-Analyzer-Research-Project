def update_dictionary(sent_dict, unknown_words, score):
    # Simple update logic: assume unknown words carry the same sentiment as the headline
    if not unknown_words:
        return sent_dict

    weight = score / len(unknown_words) if len(unknown_words) else 0

    for word in unknown_words:
        if word in sent_dict:
            # Average the existing and new score
            sent_dict[word] = (sent_dict[word] + weight) / 2
        else:
            sent_dict[word] = weight

    return sent_dict