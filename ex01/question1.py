import math

def degrees_to_radians(deg):
    return deg * math.pi / 180

degrees_list = [0, 90, 180, 45, 30, 10, 5, 1]

print("degrees,radians,sin,cos")

for deg in degrees_list:
    rad = degrees_to_radians(deg)
    sin_val = math.sin(rad)
    cos_val = math.cos(rad)
    print(f"{deg},{rad},{sin_val},{cos_val}")