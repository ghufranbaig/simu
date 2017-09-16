import random
import math
import os
import Queue as Q
from copy import deepcopy
from Fermi import Fermi
from Fermi import getCliques
from Fermi import FermiPreCompute
from Fermi import FermiPreCompute2

from helper_misc import *
import numpy as np

timesteps = 1

outputDir = "res/"

CHANNEL_BW = 5
INT = 1e-3
SNR = 3
rad = 100
bw = 100
rbgsize = 1
spectrumBW = {6:1.4e6, 15:3.0e6, 25:5.0e6, 50:10.0e6, 75:15.0e6, 100:20.0e6}
SpectralEfficiencyForCqi = [
  0.0,
  0.15, 0.23, 0.38, 0.6, 0.88, 1.18,
  1.48, 1.91, 2.41,
  2.73, 3.32, 3.9, 4.52, 5.12, 5.55
]


# Taken from Rahman's model in ns-3
Noise = 1.65959e-19
# Noise = 3.98107e-21

# Radius of an eNodeB cell, so all users are at distance [rad_i, rad_o], in meters
#rad_o = 100
#rad_i = 30

# Thresholds for calculating interference (dist_thresh is currently not used)
dist_thresh = 180
cqi_thresh = 3

def dist(a,b):
	x = a[0] - b[0]
	y = a[1] - b[1]
	return math.hypot(x,y)

def angle(a,b):
	x = a[0] - b[0]
	y = a[1] - b[1]
	return math.atan2(y,x)

def dBmtoW(dbm):
	return (10**((dbm-30)/10.0))

def WtodBm(W):
	return (10*math.log10(W) + 30)

def RxPowertoDistance(P_r,P_t):
	n = 3.5
	L_0 = 25.686
	d_0 = 1
	return( d_0 * 10**( math.log10(P_t/P_r)/n - L_0/(n*10.0) ) )

def RxPwrtoTxPower (P_r,d):
	n = 3.5
	L_0 = 25.686
	d_0 = 1
	return ( P_r*(10**((L_0 + 10*n*math.log10(d/d_0))/10.0)) )

def getOverHearingRange(enb,ue):
	d = dist(enb,ue)
	TxP = RxPwrtoTxPower (dBmtoW(-90.0),d)
	m_range = RxPowertoDistance(dBmtoW(-100.0),TxP)
	return m_range
def prop_model(d,P_t):
	f = 3.600;
	d = d/1000.0;
	h_b = 30.0;
	h_m = 5.0;

	if (d == 0):
		d = 1

	A_f = 92.4+20*math.log10(d)+20*math.log10(f);
	A_bm = 20.41+9.83*math.log10(d)+7.894*math.log10(f)+9.56*(math.log10(f))**2;
	G_t = math.log10(h_b/200)*(13.958+5.78*(math.log10(d))**2);
	G_r = (42.57+13.7*math.log10(f))*(math.log10(h_m)-0.585);

	PL = A_f + A_bm - G_t - G_r;
	P_r = P_t/(10**(PL/10))

	# Taken from Rahman's model in ns-3	
	# n = 3.5
	# L_0 = 25.686
	# d_0 = 1
	#P_r = P_t/(10**((L_0 + 10*n*math.log10(d/d_0))/10))

	#print (P_t,P_r)
	return P_r

def rcvd_pows(bs_coords,ue):
	# Taken from Rahman's model in ns-3	
	P_t = 1
	pows = {}
	dists = {}
	for i in range(len(bs_coords)):
		bs = bs_coords[i]
		d = dist(bs,ue)
		P_r = prop_model(d,P_t)
		pows[i] = P_r
		dists[i] = d
	return (pows,dists)

def rcvd_pows_bs(bs_coords,enb):
	P_t = 1
	pows = {}
	dists = {}
	curr_bs = bs_coords[enb]
	for i in range(len(bs_coords)):
		bs = bs_coords[i]
		d = dist(bs,curr_bs)
		if d == 0:
			P_r = -1
		else:
			P_r = prop_model(d,P_t)
		pows[i] = P_r
		dists[i] = d
	return (pows,dists)

def gen_eNbs_coord(n,l,w):
	coord = []
	for i in range (n):
		x=random.randint(0,l)
		y=random.randint(0,w)
		coord.append((x,y))
	return coord

def gen_ue_coord(enb,l,w,rad_i,rad_o):
	while(True):
		r = random.uniform(rad_i, rad_o)
		q = random.uniform(0, 2*math.pi)
		x = enb[0] + r*math.cos(q)
		y = enb[1] + r*math.sin(q)
		if ((x>0 and x<l) and (y>0 and y<w)):
			return(x,y)

def gen_ue_coord(n,l,w,Op):
	coord = []
	ues = 0
	while(ues < n):
		x=random.randint(0,l)
		y=random.randint(0,w)
		for e in Op.eNBs:
			if (dist(e.coords,(x,y)) < 50):
				coord.append((x,y))
				ues += 1
				break
	return coord

def max_interferer(interferers,powers):
	max_i = interferers[0]
	for i in interferers:
		if (powers[i] > powers[max_i]):
			max_i = i
	return max_i



def GenerateGraphInfo(enb_coord,UEs):
	n = len(enb_coord)
	Distance = {}
	RxPower = {}
	CQIVals = {}

	Distance_BS = {}
	RxPower_BS = {}

	for i in range(n):
		rcv_pow = []
		dists = []
		CQIs = []
		Actual_CQI = []

		class1 = [True for a in range(len(UEs[i]))]
		class1_S_E = [True for a in range(len(UEs[i]))]		


		ue_num = 0
		for u in UEs[i]:
			(powers,distances) = rcvd_pows(enb_coord,u)
			rcv_pow.append(powers)
			dists.append(distances)

			snr = {}
			cqis = {}
			for k in range(n):
				if(i==k):
					snr[k] = powers[i]/(Noise*spectrumBW[bw])
				else :
					snr[k] = powers[i]/(powers[k] + Noise*spectrumBW[bw])
				cqis[k] = getCQI(getSpectralEfficiency(snr[k]))
			CQIs.append(cqis)

		Distance[i] = dists
		RxPower[i] = rcv_pow
		CQIVals[i] = CQIs


		(powers_bs,dist_bs) = rcvd_pows_bs(enb_coord,i)

		Distance_BS[i] = dist_bs
		RxPower_BS[i] = powers_bs

	
	return (Distance,RxPower,CQIVals,Distance_BS,RxPower_BS)

def GenerateGraphInfoBS(enb_coord):
	Distance_BS = {}
	RxPower_BS = {}

	for i in enb_coord:
		(powers_bs,dist_bs) = rcvd_pows_bs(enb_coord.values(),i)
		Distance_BS[i] = dist_bs
		RxPower_BS[i] = powers_bs

	
	return (Distance_BS,RxPower_BS)

def GenerateGraph_BS(Distance_BS,RxPower_BS,enb_coord):
	IDs = enb_coord.keys()
	G = {}

	rx_power_thresh  = dBmtoW(-80)
	for i in enb_coord:
		G[i] = set()

	for i in IDs:
		for j in IDs:
			if (i==j):
				continue
			if (RxPower_BS[i][j] > rx_power_thresh):
				G[i].add(j)
				G[j].add(i)
								
	for i in G:
		G[i] = list(G[i])

	return (G)

# Find connected components of a graph
def connected_graphs(G):
	conn_G = []
	v = list(G.keys())
	q = Q.Queue()
	while (len(v)!=0):
		q.put(v[0])
		g = []
		while q.empty() == False:
			curr = q.get()
			if curr in v:
				g.append(curr)
				v.remove(curr)
			for n in G[curr]:
				if n in v:
					q.put(n)
					v.remove(n)
					g.append(n)
		G_dash = {}
		for i in g:
			G_dash[i] = deepcopy(G[i])
		conn_G.append(G_dash)
	return conn_G



def getSpectralEfficiency(snr):
	BER = 0.00005
	return math.log(1+(snr/(-math.log(5*BER)/1.5)),2)

def getCQI(s):
	cqi = 0
	for i in range(len(SpectralEfficiencyForCqi)):
		if(s>SpectralEfficiencyForCqi[i]):
			cqi+=1
		else:
			break
	return (cqi-1)


def assign_ues_to_enbs(enbs,ues):
	assign_final = {}
	for u in ues:
		enb_id = 0
		min_d = dist(u,enbs[0])
		for i in range(len(enbs)):
			d=dist(u,enbs[i])
			if (d < min_d):
				enb_id = i
				min_d = d
		assign_final[u] = enb_id
	return assign_final


def remove_node(e,c_map):
	if (e in c_map):
		for n in c_map:
			if e in c_map[n]:
				c_map[n].remove(e)
		del c_map[e]

def FermiAllocations(UEs,u_m,G,load,N,info,comp,logging=False):
		assign = []
		alloc = []
		assign_grid = {}

		for enb in UEs:
			assign_grid[enb] = [0]*N


                # Main calculation
		for c_map in G:
			(channel_assignment,allocated_share) = Fermi(c_map,load,N)
			assign.append(channel_assignment)
			alloc.append(allocated_share)
			#assign_us.append(KlogN(c_map,W,N))

		Assign = {}
		for a in assign:
			Assign.update(a)
		Allocated_share = {}
		for a in Assign:
			share = 0
			for interval in Assign[a]:
				share += interval[1]-interval[0]+1

				for j in range(interval[0],interval[1]+1):
					assign_grid[a][j] = 1

			Allocated_share[a] = share
		
		share_UE={}
		for a in Allocated_share:
			if load[a] == 0:
				share = 0
			else:
				share = eval("%.2f" % (float(Allocated_share[a])/load[a]))
			for u in UEs[a]:
				share_UE[u_m[u]]=share

		writeInfo('FERMI',Assign,Allocated_share,share_UE,info)
		#print assign_grid
		if logging:
			print 'share: ', Allocated_share

		return (Assign,Allocated_share,assign_grid)

def FermiAllocationsSimple(UEs,FermiIntfMap,load,N):
		assign = []
		alloc = []
		assign_grid = {}

		for enb in UEs:
			assign_grid[enb] = [0]*N

                # Main calculation
		for c_map in FermiIntfMap:
			(channel_assignment,allocated_share) = FermiPreCompute(c_map[0],load,N,c_map[1],c_map[2],c_map[3])
			assign.append(channel_assignment)
			alloc.append(allocated_share)
			#assign_us.append(KlogN(c_map,W,N))

		Assign = {}
		for a in assign:
			Assign.update(a)



		Alloc_actual = {}
		for a in alloc:
			Alloc_actual.update(a)




		Allocated_share = {}
		for a in Assign:
			share = 0
			for interval in Assign[a]:
				share += interval[1]-interval[0]+1

				for j in range(interval[0],interval[1]+1):
					assign_grid[a][j] = 1

			Allocated_share[a] = share
			#if (load[a] != 0 and Allocated_share[a] == 0):
			#	print a, 'Something Wrong', Alloc_actual[a]
		#info2.write(str(load))
		#info2.write('\n')
		#writeInfo('FERMI',Assign,Allocated_share,Allocated_share,info2)

		return (Assign,Allocated_share,assign_grid)
 

def getSNR(N,assignment,Rx_power,enb,j):
	SNR = []
	for i in range(N):
		if assignment[enb][i] == 0:
			snr = -1
		else:
			I = 0;
			for bs in Rx_power:
				#print assignment[bs][i], bs, enb
				if (assignment[bs][i] == 1):
					I += Rx_power[enb][j][bs]
			snr = getSpectralEfficiency(Rx_power[enb][j][enb]/((Noise*spectrumBW[bw])+I-Rx_power[enb][j][enb])) #/ getSpectralEfficiency(Rx_power[enb][j][enb]/(Noise*spectrumBW[bw]))
			#print snr
		SNR.append(snr)
	return SNR

def getSNR(ue,enb,ch):
	for bs in Rx_power:
		if (assignment[bs][ch] == 1 and activeUsersPerAP[bs] > 0):
			I += Rx_power[enb][ue][bs]
	snr = getSpectralEfficiency(Rx_power[enb][ue][enb]/((Noise*spectrumBW[bw])+I-Rx_power[enb][ue][enb]))

	return snr
#def getInterference(N,assignment,Rx_power,enb,j):

class Operator:
	def __init__(self, op_id):
		self.ID = op_id
		self.UEs = []
		self.eNBs =[]

	def add_eNBs (self, eNB_coords, offset):
		i = 0
		for e in eNB_coords:
			self.eNBs.append(eNodeB(offset + i, e, self.ID))
			i += 1
	def add_eNB (self, eNB_coord, eNB_ID):
		self.eNBs.append(eNodeB(eNB_ID, eNB_coord, self.ID))


	def add_UEs (self, UE_coords, offset):
		i = 0
		for u in UE_coords:
			self.UEs.append(UserE(offset + i, u, self.ID))
			#activity_for_ue = np.random.randint(2, size=(timesteps)).tolist()
			activity_for_ue = np.ones(timesteps).tolist()
			self.UEs[i].activity = activity_for_ue
			i += 1

	def assignUestoeNBs (self):
		ue_coords = []
		enb_coords = []
		for u in self.UEs:
			ue_coords.append(u.coords)
		for e in self.eNBs:
			enb_coords.append(e.coords)

		temp_assign = assign_ues_to_enbs(enb_coords,ue_coords)

		i = 0
		for u in ue_coords:
			assert (cmp(self.UEs[i].coords, u) == 0),"User coords don't match"

			self.UEs[i].eNBID = self.eNBs[temp_assign[u]].ID
			self.UEs[i].eNB = self.eNBs[temp_assign[u]]
			self.eNBs[temp_assign[u]].UEs.append(self.UEs[i])
			i += 1
	def generate_user_activity (self):
		for u in self.UEs:
			u.genActivity()

	def updateAtEnd():
		for e in self.eNBs:
			activeUsersPerAP[e.ID] = e.activeUEs 

	def getData(self):
		operator_enbs = {}
		operator_ues = {}
		enb_coord = []
		UEs = {}
		UE_activity = {}
		#UE_activity = {}
		for e in self.eNBs:
			enb_coord.append(e.coords)
			UEs[e.ID] = []
			UE_activity[e.ID] = []

		operator_enbs[self.ID] = enb_coord
		operator_ues[self.ID] = []
		for u in self.UEs:
			operator_ues[self.ID].append(u.coords)
			UEs[u.eNB].append(u.coords)
			UE_activity[u.eNB].append(u.activity)
		return (operator_enbs,operator_ues,UEs,enb_coord,UE_activity)



class eNodeB:

	def __init__(self, enb_id, coord, operator):
		self.ID = enb_id
		self.coords = coord
		self.operator = operator    # creates a new empty list for each dog
		self.UEs = []
		self.myChans = []
		self.sharedChans = []
		self.activeUEs = 0
		self.operationalChannel = []
		self.APonChannel = {}

	def execute(t):
		activeUEs = 0
		for u in UEs:
			rem = u.execute(self.operationalChannel,self.APonChannel,t)
			if rem > 0:
				activeUEs += 1
		self.activeUEs = activeUEs

		#activeUsersPerAP[self.ID] = activeUEs

	def add_ue(self, ue):
		self.UEs.append(ue)



class UserE:

	def __init__(self, ue_id, coord, operator):
		self.ID = ue_id
		self.coords = coord
		self.operator = operator
		self.eNB = -1
		self.activity = []
		self.totData = 0
		self.RxPower = {}
		self.distances = {}

	def assign_eNB(self, enb_id):
		self.eNB = enb_id

	def genActivity(self,size_):
		self.activity=np.random.choice([0, 1], size=(size_,), p=[4./5, 1./5])
		for i in range(size_):
			if(self.activity[i]==1):
				self.activity[i] = random.randint(5,500)
		print (self.activity)

	def updateRem(n,t):
		self.totData -= min(n,self.totData) 
		self.totData += self.activity[t]
		return self.totData

	def execute(t,operationalChannel,APonChannel):
		totData = 0
		for ch in operationalChannel:
			totUsersOnChan = 0
			for a in APonChannel[ch]:
				totUsersOnChan+=activeUsersPerAP[a]
			totData+= CHANNEL_BW*self.getSE(ch)*INT/totUsersOnChan
		return self.updateRem(totData,t)

	def getSE(ch):
		for bs in self.RxPower:
			if (assignment[bs][ch] == 1 and activeUsersPerAP[bs] > 0):
				I += self.RxPower[bs]
			se = getSpectralEfficiency(self.RxPower[self.eNB]((Noise*spectrumBW[bw])+I-self.RxPower[self.eNB]))

		return se


from itertools import izip
def genPowerInfo (OP):
	eNBs={}
	for op in OP:
		for e in op.eNBs:
			eNBs[e.ID]=e.coords
	IDs = eNBs.keys()
	enb_coord = eNBs.values()

	for op in OP:
		for u in op.UEs:
			(powers,distances) = rcvd_pows(enb_coord,u.coords)
			u.RxPower = dict(izip(IDs, powers))
			u.distances = dict(izip(IDs, distances))


def simu():
	global assignment
	global activeUsersPerAP


def main(operators,npo,usersPerOperator,N,l,w):

	print N
	Operators = []
	print(npo)
	opEnb = {}
	j = 0
	k = 0;
	for i in range(operators):
		Op = Operator(i)
		temp_coords = gen_eNbs_coord(npo[i],l,w)
		Op.add_eNBs(temp_coords,j)
		j += len(temp_coords)

		#temp_coords = gen_ue_coord(usersPerOperator[i],l,w,Op)
		#Op.add_UEs(temp_coords,k)
		#k += len(temp_coords)
		#Op.assignUestoeNBs()
		Operators.append(Op)


	enb_coord = {}
	for op in Operators:
		opEnb[op.ID] = []
		for e in op.eNBs:
			enb_coord[e.ID] = e.coords
			opEnb[op.ID].append(e.ID)

	comp = open('graph.py','w')


	(Distance_BS,RxPower_BS) = GenerateGraphInfoBS(enb_coord)
	i_map = GenerateGraph_BS(Distance_BS,RxPower_BS,enb_coord)
	G = connected_graphs(i_map)
	
	
	G_info = []
	for m in G:
		print 'graph size', len(m)
		(i_map,i_map_,fill_in,C) = getCliques(m)
		G_info.append((i_map,i_map_,fill_in,C))

	comp.write('G_info='+str(G_info)+'\n')
	comp.write('opEnb='+str(opEnb)+'\n')

	comp.write('enb_coord='+str(enb_coord)+'\n')
	comp.write('npo='+str(npo)+'\n')
	comp.write('l,w='+str(l)+','+str(w)+'\n')

	
	comp.close()

	'''
	Operators = []
	k = 0
	for i in opEnb:
		Op = Operator(i)
		for e in opEnb[i]:
			Op.add_eNB (enb_coord[e], e)
		temp_coords = gen_ue_coord(usersPerOperator[i],l,w,Op)
		Op.add_UEs(temp_coords,k)
		k += len(temp_coords)
		Op.assignUestoeNBs()
		Operators.append(Op)
	genPowerInfo (Operators)
	'''

# Body, generating scripts
#os.system('mkdir ' + outputDir)
for z in range(10):
	l = 2500
	w = 2500
	N = 2500
	#info2.write(str(z)+'\n')
	operators = 3
	npo = [1000,1000,1000]
	usersPerOperator = {0:12000,1:12000,2:12000}
	main(operators,npo,usersPerOperator,N,l,w)
	os.system('mv graph.py graph'+str(z)+'.py')

#u=UserE(1,(1,1),1)
#u.genActivity(10)
#print(getSpectralEfficiency(1000))
#A:0 B:1 C:2 D:3 E:4 F:5 G:6
'''
G = {0:[1,3],1:[2,0],2:[1,3,4,5],3:[0,2,6],4:[2,5],5:[2,4],6:[3]}
load = {0:1,1:1,2:2,3:1,4:2,5:1,6:3}
opEnb = {0:[0,1,5],1:[4,6],2:[2,3]}
(i_map,i_map_,fill_in,C) = getCliques(G)
print C
N=30
#FermiPreCompute2(i_map,load,N,i_map_,fill_in,C,opEnb)
Assign = FermiPreCompute(i_map,load,N,i_map_,fill_in,C)
print Assign
# print g_
'''
