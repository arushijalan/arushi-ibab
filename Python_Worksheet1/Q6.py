def is_prime(N):
    if N < 2:
        return False
    for i in range(2, int(N**0.5) + 1):
        if N % i == 0:
            return False
    return True

def main():
    N = int(input("Enter a number: "))
    if is_prime(N):
        print(N, "is a prime number.")
    else:
        print(N, "is a composite number.")

if __name__ == "__main__":
    main()