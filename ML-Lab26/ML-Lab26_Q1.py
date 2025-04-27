def is_positive_definite_manual(A):
    a, b = A[0]
    c, d = A[1]
    # Characteristic equation: (9-λ)(21-λ) - (-15)^2 = 0
    # Expand manually
    trace = a + d  # sum of diagonals
    det = a*d - b*c  # determinant
    
    discriminant = trace**2 - 4*det
    sqrt_disc = discriminant**0.5

    eigen1 = (trace + sqrt_disc) / 2
    eigen2 = (trace - sqrt_disc) / 2
    
    print(f"Eigenvalues are: {eigen1:.2f} and {eigen2:.2f}")
    
    return eigen1 > 0 and eigen2 > 0

def main():
    A = [[9, -15], [-15, 21]]
    result = is_positive_definite_manual(A)
    print("Matrix A is positive definite:" if result else "Matrix A is NOT positive definite.")

if __name__ == "__main__":
    main()
