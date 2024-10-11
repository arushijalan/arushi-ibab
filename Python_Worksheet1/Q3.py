def count_1s():
    binary = []
    D = int(input("Enter a decimal number that you want to convert into binary: "))
    while D > 1:
        a = D % 2
        binary.append(a)
        D = D // 2

    if D == 1 or D == 0:
        binary.append(D)

    binary.reverse()
    binary_new = ''.join(str(e) for e in binary)

    count = binary_new.count('1')
    print("The number of 1s in are", count)

def main():
    count_1s()

if __name__ == "__main__":
    main()