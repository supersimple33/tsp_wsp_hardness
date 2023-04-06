import sys

print(sys.argv)
opt = float(sys.argv[1])
cur = float(sys.argv[2])
print(round(100 * (cur - opt) / opt, 1))