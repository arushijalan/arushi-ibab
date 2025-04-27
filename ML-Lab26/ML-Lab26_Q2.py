def find_hessian_eigenvalues_manual(x, y):
    a = 12 * x**2
    d = 2
    b = c = -1
    
    trace = a + d
    det = a*d - b*c
    
    discriminant = trace**2 - 4*det
    sqrt_disc = discriminant**0.5

    eigen1 = (trace + sqrt_disc) / 2
    eigen2 = (trace - sqrt_disc) / 2

    return eigen1, eigen2

def main():
    eigen1, eigen2 = find_hessian_eigenvalues_manual(3,1)
    print(f"Eigenvalues are {eigen1:.2f} and {eigen2:.2f}")

if __name__ == "__main__":
    main()
