def replace():
    S = input("Enter a string: ")
    c = input("Enter the character you want to replace: ")
    temp1 = c
    C = c.swapcase()
    d = input("Enter the character you want to replace with: ")
    temp2 = d
    D = d.swapcase()
    S = S.replace(temp1,temp2)
    S = S.replace(C,D)
    print(S)

def main():
    replace()

if __name__ == "__main__":
    main()