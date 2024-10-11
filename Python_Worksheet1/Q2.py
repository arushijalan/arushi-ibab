def decimal_to_binary():
    binary = [ ]
    D = int(input("Enter a decimal number that you want to convert into binary: "))
    while D > 1:
        a = D%2
        binary.append(a)
        D = D // 2

    if D == 1 or D == 0:
        binary.append(D)

    binary.reverse()
    binary_new = ''.join(str(e) for e in binary)
    print("The binary conversion is", binary_new)

def main():
    decimal_to_binary()

if __name__ == "__main__":
     main()
