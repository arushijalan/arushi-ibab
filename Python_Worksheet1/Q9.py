def alternate_char():
    S = input("Enter a string: ")
    print("These are the alternate characters of", S, "are", S[::2])

def main():
    alternate_char()

if __name__ == "__main__":
    main()