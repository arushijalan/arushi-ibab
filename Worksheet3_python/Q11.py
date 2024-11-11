class FormulaError(Exception):
    pass

def interactive_calculator():
    while True:
        user_input = input("Enter a formula (e.g., 1 + 1): ")

        if user_input.lower() == "quit":
            print("Exiting calculator")
            break
        try:
            parts = user_input.split()

            if len(parts) != 3:
                raise FormulaError("Formula should be in form of a number, an operator, and a number.")
            num1 = float(parts[0])
            num2 = float(parts[2])

            operator = parts[1]
            if operator not in ('+', '-'):
                raise FormulaError("Operator must be '+' or '-'.")
            if operator == '+':
                result = num1 + num2
            elif operator == '-':
                result = num1 - num2
            print(f"Result: {result}")

        except ValueError:
            raise FormulaError("Invalid input: Numbers should be int or float values.")
        except FormulaError as fe:
            print(f"FormulaError: {fe}")
interactive_calculator()
