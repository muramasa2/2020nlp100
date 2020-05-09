x = input()
y = input()
z = input()

print(x + '時の' + y + 'は' + z)

############
# 模範解答 #
############
def generate_temp(x, y, z):
    return f"{x}時の{y}は{z}"

print(generate_temp(12, "気温", 22.4))
