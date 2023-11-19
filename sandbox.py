

s = "Whose this? Lol jk I know it's you Cute SHelly omfg ur so pretty I've missed you how are you what have you been up to how's lyric?! "
s = s.lower()

# f = ""
f = ''.join([x.upper() if i % 2 == 0 else x for i, x in enumerate(s.lower())])
# for i, x in enumerate(s):
#    if i%2 ==0:
#        f += x.upper()
#    else:
#        f += x

print(f)