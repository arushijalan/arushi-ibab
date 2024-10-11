def anagrams():
    A = input("Enter first string: ")
    B = input("Enter second string: ")

    new_A = sorted(A.lower())
    new_B = sorted(B.lower())

    if new_A == new_B:
        print(A, "and", B, "are anagrams.")
    else:
        print(A, "and", B, "are not anagrams.")

def main():
    anagrams()


if __name__ == "__main__":
    main()