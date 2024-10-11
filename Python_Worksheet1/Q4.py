def binary_to_decimal(binary):
    decimal = 0
    for i in range(len(binary)):
        bit = binary[-(i+1)]
        if bit == '1':
            decimal += 2**i
    return decimal

def main():
    binary = input("Enter a binary number: ")
    decimal = binary_to_decimal(binary)
    print("The decimal value of", binary, "is", decimal)

if __name__ == "__main__":
    main()