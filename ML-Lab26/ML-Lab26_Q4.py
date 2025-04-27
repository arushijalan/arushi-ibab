def concavity_check(x, y):
    d2f_dx2 = 6 * x
    d2f_dy2 = 12 * y
    d2f_dxdy = -1

    trace = d2f_dx2 + d2f_dy2
    det = d2f_dx2 * d2f_dy2 - (d2f_dxdy)**2

    discriminant = trace**2 - 4*det
    sqrt_disc = discriminant**0.5

    eigen1 = (trace + sqrt_disc) / 2
    eigen2 = (trace - sqrt_disc) / 2

    if eigen1 > 0 and eigen2 > 0:
        return "Strictly Convex (Minimum)"
    elif eigen1 < 0 and eigen2 < 0:
        return "Strictly Concave (Maximum)"
    else:
        return "Saddle Point"

def main():
    points = [(0,0), (3,3), (3,-3)]
    for point in points:
        nature = concavity_check(*point)
        print(f"At {point}: {nature}")

if __name__ == "__main__":
    main()
