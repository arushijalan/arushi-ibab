def find_critical_points_and_classify():
    # Given function:
    # f(x,y) = 4x + 2y - x^2 - 3y^2

    # Step 1: Find gradient components manually
    # Set gradients to zero
    
    # 4 - 2x = 0 → x = 2
    x_crit = 4 / 2  # x = 2

    # 2 - 6y = 0 → y = 1/3
    y_crit = 2 / 6  # y = 1/3

    critical_point = (x_crit, y_crit)

    # Step 2: Find Hessian manually
    # Hessian is:
    # [ -2   0 ]
    # [  0  -6 ]

    # Eigenvalues are just the diagonal entries for a diagonal matrix
    eigen1 = -2
    eigen2 = -6

    # Step 3: Classify based on eigenvalues
    if eigen1 > 0 and eigen2 > 0:
        classification = "Local Minimum"
    elif eigen1 < 0 and eigen2 < 0:
        classification = "Local Maximum"
    else:
        classification = "Saddle Point"

    return critical_point, classification

def main():
    critical_point, classification = find_critical_points_and_classify()
    print(f"Critical point is at: {critical_point}")
    print(f"Nature of the critical point: {classification}")

if __name__ == "__main__":
    main()
