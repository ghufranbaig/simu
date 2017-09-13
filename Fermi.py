import math
import networkx as nx
import Queue as Q
from copy import deepcopy
from copy import copy


max_BW_chan = 20
allocation_granularity = 5

#Tree class
class Node(object):
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)
    def print_tree(self):
	print(self.data)
	print('--------')
	for c in self.children:
		c.print_tree()


#Triangulation
def BFS(graph,start,end,q):
	#print('BFS')
        paths = []
        temp_path = [start]	
        q.put(temp_path)
        while q.empty() == False:
                tmp_path = q.get()
                last_node = tmp_path[len(tmp_path)-1]
                if last_node == end:
                    paths.append(tmp_path)
                    #print ("VALID_PATH : ",tmp_path)
                for link_node in graph[last_node]:
                    if link_node not in tmp_path:
                        new_path = []
                        new_path = tmp_path + [link_node]
                        q.put(new_path)
        return paths


def max_u(u,w):
    m = u[0]
    for i in range(1,len(u)):
        if (w[u[i]]>=w[m]):
            m = u[i]
    return m


# The algorithm to transform the inital graph into chordal graph
# as mentioned in the paper in Algorithm 1, step 1 (and describe in another paper)
# 
# Input: 
#   G : interference graph - map with keys = nodes and value = [adjacent nodes]
# Output:
#   G : chordal interf graph in the same format
#   fill_in : edges added to make the graph chordal, to be removed later

def remove_node(e,c_map):
	if (e in c_map):
		for n in c_map:
			if e in c_map[n]:
				c_map[n].remove(e)
		del c_map[e]
		#c_map[e] = set()
	return c_map

def remove_nodes(e,c_map):
	for n in c_map:
		c_map[n] = c_map[n]-e
	#del c_map[e]
	for n in e:
		del c_map[n]
	return c_map

def dfs1(graph, start):
    if len(graph) == 0:
       return set()
    visited, stack = set(), [start]
    #print '**********************************'
    #print graph
    #print start
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            #print vertex
            stack.extend(graph[vertex] - visited)
    #print visited
    #print '**********************************'
    return visited	

def dfs(graph, start, endn):
    if len(graph) == 0:
       return set()
    visited, stack = set(), [start]
    #print '**********************************'
    #print graph
    #print start
    while stack:
        vertex = stack.pop()
        if vertex == endn:
        	return True
        if vertex not in visited:
            visited.add(vertex)
            #print vertex
            stack.extend(graph[vertex] - visited)
    #print visited
    #print '**********************************'
    return False	

def Triangulation (G):

    F = set()
    w = {}
    alpha = {}
    un_numbered = []
    for v in G:
        w[v] = 0
        un_numbered.append(v)
    n = len(G)
    #print n

    G_unnumbered = deepcopy(G)
    G_dashes = {}
    for node in G_unnumbered:
        G_unnumbered[node] = set(G_unnumbered[node])
    #for node in G_unnumbered:
    #    G_dashes[node] = deepcopy(G_unnumbered)

    for i in range (n,0,-1):
        print i
        z = max_u(un_numbered,w)
        alpha[z] = i
        un_numbered.remove(z)
        wz_minus = copy(w)#deepcopy(w)
        for y in un_numbered:
           #print '\t', y
           if z in G[y]:
           	w[y] += 1
           	continue
           G_dash = deepcopy(G_unnumbered)# G_dashes[y]
           
           nodes_to_remove = set()
           nodes_to_add = set()
           for node in un_numbered:
              if (node != z and node != y and wz_minus[node] >= wz_minus[y]):
              	nodes_to_remove.add(node)
		#G_dash = remove_node(node,G_dash)

           #G_dash = {}
           #for node in nodes_to_add:
           #	G_dash[node] = G_unnumbered[node] - nodes_to_remove
           G_dash = remove_nodes(nodes_to_remove,G_dash)
	   
           '''
           p = dfs1(G_dash,z)
           #print p
           if y in p :
                w[y] += 1
                F.add((y,z))
            '''

           if dfs(G_dash,z,y) :
                w[y] += 1
                F.add((y,z))
           
           #G_dashes[y] = remove_node(z,G_dash)

        G_unnumbered = remove_node(z,G_unnumbered)
        #print(z,w)

    for a in G:
        G[a]=set(G[a])

    fill_in = []
    for edge in F:
	if edge[1] not in G[edge[0]]:
        	G[edge[0]].add(edge[1])
        	G[edge[1]].add(edge[0])
		fill_in.append(edge)
    for a in G:
        G[a]=list(G[a])
    print fill_in
    return (G,fill_in)


def Triangulation2 (G):
    path_queue = Q.Queue();
    F = set()
    w = {}
    alpha = {}
    un_numbered = []
    for v in G:
        w[v] = 0
        un_numbered.append(v)
    n = len(G)
    for i in range (n,0,-1):
        #print (w)
        z = max_u(un_numbered,w)
        #print(z)
        alpha[z] = i
        un_numbered.remove(z)
        wz_minus = deepcopy(w)
        for y in un_numbered:
           p = BFS(G,y,z,path_queue)

           
           inc = False
           for x in p:
               flag = True  
               invalid_path = False             
               if (len(x) > 2):
                    for j in range (1,len(x)-1):
                       if (x[j] not in un_numbered):
                           invalid_path = True
                           flag = False
                           break
                       if (wz_minus[x[j]] >= wz_minus [y]):
                           flag = False
                           break
               if invalid_path :
                       continue
               if flag :
                       inc = True
                       break
           if inc :
                w[y] += 1
                F.add((y,z))
        #print(z,w)


    for a in G:
        G[a]=set(G[a])

    fill_in = []
    for edge in F:
	if edge[1] not in G[edge[0]]:
        	G[edge[0]].add(edge[1])
        	G[edge[1]].add(edge[0])
		fill_in.append(edge)
    for a in G:
        G[a]=list(G[a])
    return (G,fill_in)



# Allocation
def s_tuple(v_i,C,L,cliqueAssoc):
        t_i = 0
        l_i = -1
        for j in cliqueAssoc[v_i]:
                t_i += 1
                if L[j] > l_i:
                        l_i = L[j]
        return (l_i,t_i)

def s_tuple2(v_i,C,L):
        t_i = 0
        l_i = -1
        for j in range (len(C)):
                if v_i in C[j]:
                        t_i += 1
                        if L[j] > l_i:
                                l_i = L[j]
        return (l_i,t_i)

def init_alloc2 (v_i,C,l,R,cliqueAssoc):

        assigned = False
        for j in range (len(C)):
                l_sum = 0
                if v_i in C[j]:
                        for k in C[j]:
                                l_sum += l[k]

			if (l_sum > 0): # this check added to incorporate nodes with share = 0
				new_val = math.floor(l[v_i]*R[j]/l_sum + 0.5)
			else:
                        	new_val = 0
                        if  not assigned:
                                A = new_val
                                assigned = True
                        elif new_val < A:
                                A = new_val
        return A

def init_alloc (v_i,C,l,R,cliqueAssoc,L):

        assigned = False
        for j in cliqueAssoc[v_i]:
                l_sum = L[j]

		if (l_sum > 0): # this check added to incorporate nodes with share = 0
			new_val = math.floor(l[v_i]*R[j]/l_sum + 0.5)
		else:
                	new_val = 0
                if  not assigned:
                        A = new_val
                        assigned = True
                elif new_val < A:
                        A = new_val
        return A

def max_rank(s):
        max_r = 0
        for i in range (1,len(s)):
                if (s[i][0]>s[max_r][0]):
                        max_r = i
                elif (s[i][0]==s[max_r][0]):
                        if (s[i][1]>s[max_r][1]):
                                max_r = i
        return max_r
        
      
def cliques(graph):
	G=nx.Graph()
	for n in graph:
		G.add_node(n)
	for n in graph:
		for e in graph[n]:
			G.add_edge(n,e)


	C = list(nx.chordal_graph_cliques(G))
	for i in range (len(C)):
		C[i] = list(C[i])
	return C



def getMaxRank(U,C,L,cliqueAssoc):
	max_r = 0
	s = []
        for j in range(len(U)):
		i = U[j]
                s.append(s_tuple(i,C,L,cliqueAssoc))
		#print s
		if (s[j][0]>s[max_r][0]):
                        max_r = j
                elif (s[j][0]==s[max_r][0]):
                        if (s[j][1]>s[max_r][1]):
                                max_r = j
	return max_r

# Calculate Fermi allocation as described in the paper as Algorithm 2
# and also the first part of step 2 of Algorithm 1 in the paper
# Input:
#   G - chordal graph -
#   load - map (node, #users)
#   N - number of sub-channels
#   C - set of maximal cliques, array of cliques, each being an array of belonging nodes
# Output:
#   Alloc - share per eNodeB, map (eNodeB_ID, share)
def Allocate (G,load,N,C):
        V = list(G.keys())
        U = V[:]
        A = {}
        L = []
        R = []
        for j in range(len(C)):
                sum_load = 0
                for i in C[j]:
                        sum_load += load[i]
                L.append(sum_load)
                R.append(N)
        
	cliqueAssoc = {}
	for i in range(len(C)):
		for v in C[i]:
			if v in cliqueAssoc:
				cliqueAssoc[v].append(i)
			else:
				cliqueAssoc[v] = [i]

	maxRank = getMaxRank(U,C,L,cliqueAssoc)
        #for i in U:
        #        A[i]=init_alloc (i,C,load,R,cliqueAssoc)
        
        numm = 0
        Alloc = {}
        while len(U) > 0:
                #print len(U)
                v_0 = U[maxRank]
                A[v_0] = init_alloc (v_0,C,load,R,cliqueAssoc,L)
		#print (A)
                Alloc[v_0]=int(A[v_0])
                U.remove(v_0)

                #next_max_rank = getMaxRank(U,C,L,cliqueAssoc)
                for j in cliqueAssoc[v_0]:
                                R[j] -= A[v_0]
                                C[j].remove(v_0)
                                L[j] -= load[v_0]
                
                maxRank = getMaxRank(U,C,L,cliqueAssoc)

                #if (maxRank != next_max_rank):
                    #print 'error!!!'
                #    numm += 1
                                
                #for i in U:
                #        A[i]=init_alloc (i,C,load,R,cliqueAssoc)
                
        #print 'times err occurred:', numm
        return Alloc


# Assignment & Restoration

def find_unalloc_block(sub_channels,i):
	for n in range (i,len(sub_channels)):
		if sub_channels[n] == 1:
			return (i,n-1)
	return (i,len(sub_channels)-1)

def find_unalloc_block2(sub_channels,i,j):
	for n in range (i,j+1):
		if sub_channels[n] == 1:
			return (i,n-1)
	return (i,j)

def find_unalloc(sub_channels,i):
	for n in range (i,len(sub_channels)):
		if sub_channels[n] == 0:
			return n
	return -1


def get_max_blk(blks_avail):
	blk = blks_avail[0]
	for block in blks_avail:
		if (block[1]-block[0]) > (blk[1]-blk[0]) :
			blk = block
	return blk
	

def allocate_sub_chan(v,sub_channels,alloc):

	start = find_unalloc(sub_channels,0)
	blks_avail = []
	while (start != -1) :
		blk = find_unalloc_block(sub_channels,start)
		if (blk[1]-blk[0]+1) >= alloc :
			return [(blk[0],blk[0]+alloc-1)]
		blks_avail.append(blk)
		start = find_unalloc(sub_channels,blk[1]+1)
	rem = alloc
	assigned = []
	while not (rem == 0 or len(blks_avail)==0):
		blk = get_max_blk(blks_avail)
		if (blk[1]-blk[0]+1 < rem):
			assigned.append(blk)
			blks_avail.remove(blk)
			rem -= blk[1]-blk[0]+1
		else:
			assigned.append((blk[0],blk[0]+rem-1))
			rem = 0
			break
	return assigned

def extend_blk(blk,avail,alloc):
    alloc -= blk[1]-blk[0]+1
    extendable = False
    st = blk[0]
    end = blk[1]

    idx = blk[0]-1
    while (idx >= 0 and avail[idx]==0 and alloc > 0):
        avail[idx] = 1
        extendable = True
        alloc -= 1
        idx -= 1
    if (extendable):
        st = idx+1

    extendable = False
    idx = blk[1]+1
    while (idx < len(avail)  and avail[idx]==0 and alloc > 0):
        avail[idx] = 1
        extendable = True
        alloc -= 1
        idx += 1
    if (extendable):
        end = idx-1
    return (st,end)

def allocate_pref_sub_chan(v,sub_channels,alloc,avail):

    start = find_unalloc(sub_channels,0)
    blks_avail = []
    while (start != -1) :
        blk = find_unalloc_block(sub_channels,start)
        if (blk[1]-blk[0]+1) >= alloc :
            return [(blk[0],blk[0]+alloc-1)]
        blks_avail.append(blk)
        start = find_unalloc(sub_channels,blk[1]+1)
    rem = alloc
    assigned = []
    while not (rem == 0 or len(blks_avail)==0):
        blk = get_max_blk(blks_avail)
        if (blk[1]-blk[0]+1 < rem):
            blks_avail.remove(blk)
            blk = extend_blk(blk,avail,rem)
            assigned.append(blk)
            rem -= blk[1]-blk[0]+1
        else:
            assigned.append((blk[0],blk[0]+rem-1))
            rem = 0
            break
    return assigned


def allocate_adj_subchan():
    max_block_size = 30
    for blk in strt_points:
	rev = blk[0]
	fwd = blk[1]
	if (fwd-rev+1 >= max_block_size):
		continue
	avail_blk = []
	flag = False
	end = 0
	for i in range(rev-1,-1,-1):
		if (subchannel[i]==1):
			break
		flag = True
		end = i
		subchannel[i] = 1
	if (flag):
		avail_blk.append((rev-1,end))
		blk_size = avail_blk[-1][0]-avail_blk[-1][1]+1
		if (blk_size == rem or blk_size - rem >= max_block_size):
			#assign and return
			rem 
		elif (blk_size > rem):
			remainders [len(avail_blk)-1] = blk_size - rem
			rem
	
	flag = False
	end = 0
	for i in range(fwd+1,len(subchannel)):
		if (subchannel[i]==1):
			break
		flag = True
		end = i
		subchannel[i] = 1
	if (flag):
		avail_blk.append((fwd+1,end))
			



def allocate_sub_chan2(v,sub_channels,alloc,prefSubchannels):

    #if (v==4):
    #	print prefSubchannels 
    sub_channels2 = sub_channels[:]
    assign1 = allocate_pref_sub_chan(v,prefSubchannels,alloc,sub_channels)
    sub_channels2 = color_channel(assign1,sub_channels2)
        
    #if (v == 4):
    #    print sub_channels
    #    print assign1
    for i in range(len(assign1)):
        blk = assign1[i]
        alloc -= blk[1]-blk[0]+1

    if (alloc <= 0):
        return assign1

    allocate_adj_subchan()

    assign2 = allocate_sub_chan(v,sub_channels2,alloc)
    for blk in assign2:
    	assign1.append(blk)
    return assign1
    	

def color_channel(assigned,subchannels,clr=1):
	clrd = 0
	for block in assigned:
		for i in range(block[0],block[1]+1):
			subchannels[i] = clr
			clrd += 1
	return subchannels

def makeTree(root,C):
	children = []
	for c in C:
		if len(set(c) & set(root.data)) > 0:
			children.append(c)
	for ch in children:
		C.remove(ch)
	for ch in children:
		root.add_child(makeTree(Node(ch),C))
	return root



# Second part of step 2 of Algorithm 1 in Fermi, summarized in the text
# Assignes the actual channels based on previously calculated shares
# Input:
#   Allocation - from the previous step
#   N - number of channels
#   C - cliques
# Output:
#   Assign - map (nodeID, [channels])
def Assignment(Alloc,N,C):
	
	root = Node(C[-1])
	C.remove(C[-1])
	tree = makeTree(root,C)	
	
	Assign = {}
	subchannels = [0 for i in range(N)]
	q = Q.Queue()
	q.put(tree)
	U = list(Alloc.keys())
	while (q.empty() == False):
		curr_node = q.get()
		subchannels = [0 for i in range(N)]
		for ch in curr_node.children:
			q.put(ch)


		#Alc = 0
		for v in curr_node.data:
			if v in Assign:
				subchannels = color_channel(Assign[v],subchannels)
				#alc = 0
				#for blk in Assign[v]:
				#	alc += blk[1]-blk[0]+1
				#if (alc > Alloc[v]):
				#	print v, 'allocated more', Alloc[v], alc
				#Alc += alc
		

		for v in curr_node.data:
			if v in U:
				Assign[v]=allocate_sub_chan(v,subchannels,Alloc[v]) 

				U.remove(v)
				subchannels = color_channel(Assign[v],subchannels)
	return Assign


def Assignment3(Alloc,N,C,i_map,opEnbs):

    C1= copy(C)
    eNBtoOp = {}
    for e in opEnbs:
        for n in opEnbs[e]:
            eNBtoOp[n] = e

    Assign = {}
    U = list(Alloc.keys())
    cliqueAssoc = {}
    for i in range(len(C)):
        for v in C[i]:
            if v in cliqueAssoc:
                cliqueAssoc[v].append(i)
            else:
                cliqueAssoc[v] = [i]

    prefSubchannels = {}
    for e in U:
    	prefSubchannels[e] = [1 for i in range(N)]

    availSubchannels = {}
    for e in U:
        availSubchannels[e] = [0 for i in range(N)]

    starting_points = {}
    for e in U:
        starting_points[e] = []

    root = Node(C1[-1])
    C1.remove(C1[-1])
    tree = makeTree(root,C1) 
    q = Q.Queue()
    q.put(tree)

    while (q.empty() == False):

        curr_node = q.get()
        for ch in curr_node.children:
            q.put(ch)

        for v in curr_node.data:

            if v not in U:
                continue
            print v
            op = eNBtoOp[v]
            subchannels = availSubchannels[v]
            Assign[v]=allocate_sub_chan2(v,subchannels,Alloc[v],prefSubchannels[v])

            for n in opEnbs[op]:
                prefSubchannels[n] = color_channel(Assign[v],prefSubchannels[n],0)
                
            for i in cliqueAssoc[v]:
                for n in C[i]:
            	    availSubchannels[n] = color_channel(Assign[v],availSubchannels[n])
            	    prefSubchannels[n] = color_channel(Assign[v],prefSubchannels[n])
                    if eNBtoOp[v] == eNBtoOp[n]:
                        for blk in Assign[v]:
                            starting_points[n].append(blk)
            U.remove(v)

    return Assign


def Assignment2(Alloc,N,C,i_map,eNBtoOp):
	
	root = Node(C[-1])
	C.remove(C[-1])
	tree = makeTree(root,C)	
	
	Assign = {}
	subchannels = [0 for i in range(N)]
	q = Q.Queue()
	q.put(tree)
	U = list(Alloc.keys())

	prefSubchannels = {}
	for e in U:
		prefSubchannels[e] = [0 for i in range(N)]

	while (q.empty() == False):
		curr_node = q.get()
		subchannels = [0 for i in range(N)]
		for ch in curr_node.children:
			q.put(ch)


		#Alc = 0
		for v in curr_node.data:
			if v in Assign:
				subchannels = color_channel(Assign[v],subchannels)
				#alc = 0
				#for blk in Assign[v]:
				#	alc += blk[1]-blk[0]+1
				#if (alc > Alloc[v]):
				#	print v, 'allocated more', Alloc[v], alc
				#Alc += alc
		

		for v in curr_node.data:
			if v in U:
				Assign[v]=allocate_sub_chan2(v,subchannels,Alloc[v],prefSubchannels[v]) 
				U.remove(v)

				for n in i_map[v]:
					if (eNBtoOp[v] == eNBtoOp[n]):
						for e in i_map[v]:
							if (eNBtoOp[e] != eNBtoOp[n]):
								prefSubchannels[e] = color_channel(Assign[v],prefSubchannels[e])

				subchannels = color_channel(Assign[v],subchannels)
	return Assign




def restore(v,free,subchannels):
	rec_blks = []
	for blk in free:
		start = find_unalloc(subchannels,blk[0])
		while(start <= blk[1] and start > -1):
			b = find_unalloc_block2(subchannels,start,blk[1])
			rec_blks.append(b)
			color_channel([b],subchannels)
			start = find_unalloc(subchannels,b[1]+1)
	return rec_blks
		

# Removes edges added to make the graph chordal
# Step 3 in Algorithm 1, and explained in text
# Input:
#   Assign - assignment
#   fill_in - added edges
#   i_map - original interference map
#   N - number of sub-channels
# Output:
#   Assign - updated assignment
def Restoration(Assign,fill_in,i_map,N):
	for e in fill_in:
		# for e[0]
		subchannels = [0 for i in range(N)]
		subchannels = color_channel(Assign[e[0]],subchannels)
		for v in i_map[e[0]]:
			subchannels = color_channel(Assign[v],subchannels)
		restored_bw = restore(e[0],Assign[e[1]],subchannels) 
		if len(restored_bw) != 0:
			for b in restored_bw:
				Assign[e[0]].append(b)
		# for e[1]
		subchannels = [0 for i in range(N)]
		subchannels = color_channel(Assign[e[1]],subchannels)
		for v in i_map[e[1]]:
			subchannels = color_channel(Assign[v],subchannels)
		restored_bw = restore(e[1],Assign[e[0]],subchannels) 
		if len(restored_bw) != 0:
			for b in restored_bw:
				Assign[e[1]].append(b)
	return Assign


def CanTakeChannel(ch,candidate,G,curr):
	#print G[candidate]
	#print curr
	for n in G[candidate]:
			for interval in curr[n]:
				if (ch>=interval[0] and ch<=interval[1]):
					return False
	return True

def freeSC(subchannels):
	free = []
	for i in range(len(subchannels)):
		if (subchannels[i] == 0):
			free.append(i)
	return free
# reclaim sub channels that are unused in a clique
# this can happen in interference map of UEs with class 1 clients
def ReclaimSC(Assign,N,C,IsClass1,i_map):
	root = Node(C[-1])
	C.remove(C[-1])
	tree = makeTree(root,C)	
	
	Reclaim = deepcopy(Assign)
	
	subchannels = [0 for i in range(N)]
	q = Q.Queue()
	q.put(tree)
	#U = list(Alloc.keys())
	while (q.empty() == False):
		curr_node = q.get()
		subchannels = [0 for i in range(N)]
		for ch in curr_node.children:
			q.put(ch)
		for v in curr_node.data:
			if v in Reclaim:
				subchannels = color_channel(Reclaim[v],subchannels)

		class1ForthisClique = []
		for v in curr_node.data:
			if (IsClass1[v]):
				class1ForthisClique.append(v)
		if (sum(subchannels) > 0 and len(class1ForthisClique) > 0): # some sub channels are free in this clique
			availableSC = freeSC(subchannels)
			num = 0
			for ch in availableSC:
				for count in range (len(class1ForthisClique)):
					if (CanTakeChannel(ch,class1ForthisClique[num],i_map,Reclaim)):
						Reclaim[class1ForthisClique[num]].append((ch,ch))
						num += 1
						num %= len(class1ForthisClique)
						break

	return Reclaim
from timeit import default_timer as timer

def getCliques(i_map):
	start = timer()
	(i_map_,fill_in) = Triangulation (deepcopy(i_map))
	end = timer()
	print 'triangulation', end - start

	C = cliques(i_map_)
	#for c in C:
	#	c.sort()
	#C.sort()
	return (i_map,i_map_,fill_in,C)
# Main function




def FermiPreCompute2(i_map,load,N,i_map_,fill_in,C,optoEnb):
    #print (C)
    start = timer()
    Alloc = Allocate(i_map_,load,N,deepcopy(C))
    end = timer()
    print 'Allocation', end - start
    print Alloc
    start = timer()
    Assign = Assignment3(Alloc,N,C,i_map,optoEnb)
    #Assign = Assignment(Alloc,N,C)
    end = timer()
    print 'Assignment', end - start
    Res = Assign

    print Assign

    shre = {}
    for n in Assign:
        shre[n] = 0
        for blk in Assign[n]:
            shre[n]+=blk[1]-blk[0]+1
        if (shre[n] != Alloc[n]):
            print 'wth bro!!'

    #Restoration(Assign,fill_in,i_map,N)
    return (Res,Alloc)

def FermiPreCompute(i_map,load,N,i_map_,fill_in,C):
	#print (C)
	start = timer()
	Alloc = Allocate(i_map_,load,N,deepcopy(C))
	end = timer()
	print 'Allocation', end - start


	#print(Alloc)
	start = timer()
	Assign = Assignment(Alloc,N,deepcopy(C))
	end = timer()
	print 'Assignment', end - start

	Res = Restoration(Assign,fill_in,i_map,N)
	#Res = Assign
	print ''
	return (Res,Alloc)

def Fermi(i_map,load,N):

	(i_map_,fill_in) = Triangulation (deepcopy(i_map))
	C = cliques(i_map_)
	for c in C:
		c.sort()
	C.sort()

	#print (C)
	Alloc = Allocate(i_map_,load,N,deepcopy(C))

	#print(Alloc)
	Assign = Assignment(Alloc,N,deepcopy(C))

	Res = Restoration(Assign,fill_in,i_map,N)

	return (Res,Alloc)




'''
import sys
sys.path.insert(0, './Balanced_Graph_Partitioning')
from timeit import default_timer as timer
import random
from networkx.algorithms.connectivity import minimum_st_node_cut
from  Balancer_Cut import *

def cutEdges(G,bestP):
    edgesremoved = set()
    effectedNodes = set()
    for p in bestP:
        par = set(p.keys()) #nx.convert.to_dict_of_dicts(p)
        for n in par:
            remEdge = set(G[n]) - par
            if (len(remEdge) == 0):
                continue
            effectedNodes.add(n)
            for node in remEdge:
                edgesremoved.add((n,node))
                effectedNodes.add(node)
    print 'effected nodes:',len(effectedNodes)
    print 'edges removed:',len(edgesremoved)/2




def getCut(graph):
    if (len(graph) < 10):
        return graph
    print 'graph size', len(graph)
    G=nx.Graph()
    for n in graph:
        G.add_node(n)
    for n in graph:
        for e in graph[n]:
            G.add_edge(n,e)

    for (u, v) in G.edges():                            # initialize the weight of the edges of the graph
            G.edge[u][v]['weight'] = 1 #random.randint(0, 10)
    for n in G.nodes():                                     # initialize the weight of the nodes of the graph
        G.node[n]['weight'] = 1 #random.random() * 100
    k = 2
    epsilon = 1.0
    n = G.number_of_nodes()
    print 'epsilon=', epsilon,'n=',n
    partitor = GraphPartitioning(epsilon, n, k, G)      # create the object and pass it the intial graph
    bestP = partitor.run()
    for el in bestP:
        print 'partition len:',len(el)
        #print el                                 # run the algorithm and return the best paritition
    cutEdges(G,bestP)
    #print("The best parition is:")
    #partitor.printPartition(bestP)
    #node_cut = nx.minimum_node_cut(G)
    #print 'min_node',node_cut
    #node_cut = nx.all_node_cuts(G)
    print("The cost of the parition is:")
    print(partitor.getCostPartition(bestP))
    
    s = graph.keys()[random.randint(0,len(graph)-1)]
    t = graph.keys()[random.randint(0,len(graph)-1)]
    node_cut = minimum_st_node_cut(G,s,t)
    
    #for n in node_cut:
    #    print n

    new_graph = copy(graph)
    #for n in node_cut:
    #    new_graph = remove_node(n,new_graph)
    
    return new_graph
'''
