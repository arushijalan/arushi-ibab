def mode():
    S = input("Enter a string: ")
    highest_freq = 0
    mode = " "
    for char in S:
        count = S.count(char)
        if count > highest_freq:
            highest_freq = count
            mode = char
        elif count == highest_freq and char not in mode:
            mode.append(char)
    print("The character with the highest frequency is", mode)

def main():
    mode()

if __name__ == "__main__":
    main()