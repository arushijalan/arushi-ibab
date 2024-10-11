def leading_whitespace():
    S = input("Enter a string with leading whitespaces: ")
    modified_S = S.lstrip()
    print("The modified string is", modified_S)
def main():
    leading_whitespace()

if __name__ == "__main__":
    main()