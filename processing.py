a = open('temp111.txt','r')

for i in range(50):
	a.readline()
	util = []
	for j in range(5):
		a.readline()
		a.readline()
		l = a.readline()
		l = l.strip()
		l = l.strip('}')
		l = l.strip('{')
		#print l
		e = l.split(',')
		temp = []
		for v in e:
			u = eval(v.split(':')[-1])
			temp.append(u)
		util.append(temp)
		a.readline()
	impr = []
	print util
	for k in range(3):
		imp = 100*(util[k+1][k]-util[0][k])/abs(util[0][k])
		impr.append(imp)
	print impr


