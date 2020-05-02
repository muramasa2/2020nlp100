s = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
s = s.strip(',').strip('.')
s_list = s.split(' ')

s_list.sort(reverse=True, key=len)

print(s_list)
