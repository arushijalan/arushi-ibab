def half_string():
    S = input("Enter a string: ")
    n = len(S)
    half = n//2
    print("Half of the string is", S, "is", S[: half])

def main():
    half_string()

if __name__ == "__main__":
    main()