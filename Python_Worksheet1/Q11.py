def first_occurrence():
    S = input("Enter a string: ")
    c = input("Enter the character whose occurrence you want to check: ")
    temp = c.lower()

    for i in range(len(S)):
        if S[i].lower() == temp:
            print("The character", c, "first occurred at index", i)
            break
    else:
        print("The character", c, "does not occur in", S)

def main():
    first_occurrence()

if __name__ == "__main__":
    main()
