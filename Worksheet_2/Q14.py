def remove_duplicate_words():
    sentence = input("Enter a sentence: ")
    print("Original sentence:", sentence)

    words = sentence.split()
    
    unique_words = {}
    
    for word in words:
        lower_word = word.lower()  
        if lower_word not in unique_words:
            unique_words[lower_word] = word  
    
    result = " ".join(unique_words.values())
    print("Sentence without duplicate words: ", result) 

remove_duplicate_words() 

