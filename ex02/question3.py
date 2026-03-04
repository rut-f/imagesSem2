#אינטרפולציה ליניארית-חד ממדית
def lerp(A, B, t):
    return (1 - t) * A + t * B
#אינטרפולציה ביליניארית-דו ממדית
def bilinear_interpolation(I00, I01, I10, I11, alpha, beta):
    return ((1 - alpha) * (1 - beta) * I00 +
            alpha * (1 - beta) * I01 +
            (1 - alpha) * beta * I10 +
            alpha * beta * I11)
#דוגמא לשימוש
I00 = 10
I01 = 20
I10 = 30
I11 = 40

alpha = 0.3  # ציר x
beta = 0.6   # ציר y

value = bilinear_interpolation(I00, I01, I10, I11, alpha, beta)
print(value)