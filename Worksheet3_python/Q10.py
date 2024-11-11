def division(a, b):
    try:
        # Perform division
        result = a / b
    except ZeroDivisionError:
        # Handle division by zero
        print("Error: Cannot divide by zero.")
    except ValueError:
        # Handle invalid input values (e.g., non-numeric)
        print("Error: Invalid input, please enter numbers.")
    except Exception as e:
        # Handle any other unknown errors
        print(f"An unknown error occurred: {e}")
    else:
        # If no exceptions occurred, print the result
        print(f"The result of {a} divided by {b} is {result}.")
    finally:
        # Always executes no matter what
        print("Division operation complete.")

# Testing the division function
division(10, 2)  # Valid division
division(10, 0)  # Division by zero
division("10", 2)  # Invalid input
division(10, "a")  # Invalid input
