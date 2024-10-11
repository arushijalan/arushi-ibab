def sum_of_squares():
    n = int(input("Enter the number till which you want to find sum of squares: "))
    result = (n*(n+1)*(2*n+1)/6) #formula of sum of squares
    print("The sum of squares till", n,"is: ", result)

def main():
    sum_of_squares()

if __name__ == "__main__":
    main()