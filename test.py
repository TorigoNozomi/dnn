NLAYERS = 5

for i in range (NLAYERS):
    exec('print(' + str(i+1) + ')')
    exec('x_' + str(i+1) + '= i+1')

exec('y = x_' + str(NLAYERS))

print(y)