#Stochastic Unit Commitment Implementation (Final Project)
#
#Bryan Arguello and Prateek Srivastava
#
#Formulation and Data Taken From "A Computationally Efficient Mixed-Integer Linear
#Formulatiion for the Thermal Unit Commitment Problem"

import cplex
import csv
import math
import numpy as np
from scipy.stats import norm

global final_iter
global Kt,J,K,OverBarP,UnderBarP,UT,DT,inistate,a,b,c,hc,cc,tcold,D
global C,G,L,ND,R,RD,RU,S0,SD,SU,U0,V0
global types, rho, scenarios, num_scenarios, T, best_guess, weight, solutions, problem
global ones, pnames, overbarpnames, Kdet, Kstoch, eps

def get_random_wind(load):
	load=load*40
	alpha1=0.0163
	beta1=0.9795
	sigma=0.2349
	mu=10.4747

	z1=norm.ppf((norm.cdf((math.log(load)-mu)/sigma)-alpha1)/beta1)
	w1=z1

	w2=np.random.normal(0,1)
	sgz=-0.1981
	z2=sgz*w1+math.sqrt(1-sgz*sgz)*w2
	phi_z2=norm.cdf(z2)

	alpha2=0.0137
	beta2=0.9863
	gamma=0.5086
	lbda=7913.19
	delta=0.7260
	zi=-177.96
	t=math.exp(norm.ppf(phi_z2*beta2+alpha2)-gamma)/delta
	wind=zi+lbda*t/(t+1)
    	
	return wind/40

def initiate_parameters():
	print "Initiating Parameters"
	global final_iter
	global Kt,J,K,OverBarP,UnderBarP,UT,DT,inistate,a,b,c,hc,cc,tcold,D
	global C,G,L,ND,R,RD,RU,S0,SD,SU,U0,V0
	global types, rho, scenarios, num_scenarios, T, best_guess, weight, solutions, problem
	global ones, pnames, overbarpnames, Kdet, Kstoch, eps 
	numunits = 10
	T = 6
	num_scenarios = 2
	J = range(1,numunits+1)
	K = range(1,T+1)
	Kt = {}
	Kdet = range(1, int(math.floor(T/2)+1))
	Kstoch = range(len(Kdet)+1, T+1)
	scenarios = range(1,num_scenarios+1)
	types = ["cd", "cu", "p", "OverBarp","v"]
	solutions = {}
	eps = 2
	final_iter = 0
	#these are given in the data set	
	rho = {}
	weight = {}
	best_guess = {}
	problem = [None for w in range(num_scenarios+1)]
	OverBarP = [None for i in range(numunits+1)]
	UnderBarP = [None for i in range(numunits+1)]
	UT = [None for i in range(numunits+1)]
	DT = [None for i in range(numunits+1)]
	inistate = [None for i in range(numunits+1)]
	a = [None for i in range(numunits+1)]
	b = [None for i in range(numunits+1)]
	c = [None for i in range(numunits+1)]
	hc = [None for i in range(numunits+1)]
	cc = [None for i in range(numunits+1)]
	tcold = [None for i in range(numunits+1)]
	D = {}
	R = {}

	#we are making these up from what we think they should be
	C = [None for i in range(numunits+1)]
	G = [None for i in range(numunits+1)]
	L = [None for i in range(numunits+1)]
	ND = [None for i in range(numunits+1)]
	RD = [None for i in range(numunits+1)]
	RU = [None for i in range(numunits+1)]
	S0 = [None for i in range(numunits+1)]
	SD = [None for i in range(numunits+1)]
	SU = [None for i in range(numunits+1)]
	U0 = [None for i in range(numunits+1)]
	V0 = [None for i in range(numunits+1)]

	#these are necessary for building the model
	ones = [1 for i in J]
	pnames = {}
	overbarpnames = {}
	for scenario in scenarios:
		for k in K:
			pnames[(k,scenario)] = ['p'+str(j) + ',' + str(k) + ',' + str(scenario) for j in J]
			overbarpnames[(k,scenario)] = ['OverBarp'+str(j)+','+str(k)+','+str(scenario) for j in J]

print "Loading Parameters with Data"
initiate_parameters()
def load_parameters():
	global final_iter
	global Kt,J,K,OverBarP,UnderBarP,UT,DT,inistate,a,b,c,hc,cc,tcold,D
	global C,G,L,ND,R,RD,RU,S0,SD,SU,U0,V0
	global types, rho, scenarios, num_scenarios, T, best_guess, weight, solutions, problem
	global ones, pnames, overbarpnames, Kdet, Kstoch, eps 
	data1 = []
	data2 = []
	data3 = []
	data4 = []
	datafile1 = "./PH2data1.csv"
	datafile2 = "./PH2data2.csv"
	datafile3 = "./PH2data3.csv"
	datafile4 = "./PH2unitdata.csv"

	with open(datafile1, 'r') as file:
		reader = csv.reader(file)
		for row in reader:
			data1.append(row)
	with open(datafile2, 'r') as file:
		reader = csv.reader(file)
		for row in reader:
			data2.append(row)
	with open(datafile3, 'r') as file:
		reader = csv.reader(file)
		for row in reader:
			data3.append(row)
	with open(datafile4, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                        data4.append(row)

	for index,row in enumerate(data1):
		index+=1
		OverBarP[index] = float(row[0])
		UnderBarP[index] = float(row[1])
		UT[index] = int(row[2])
		DT[index] = int(row[3])
		inistate[index] = int(row[4])
	
	for index2,row in enumerate(data2):
		index2+=1
		a[index2] = float(row[0])
		b[index2] = float(row[1])
		c[index2] = float(row[2])
		hc[index2] = float(row[3])
		cc[index2] = float(row[4])
		tcold[index2] = float(row[5])
	
	for index3,row in enumerate(data3):
		index3+=1
		for scenario in scenarios:
			D[(index3,scenario)] = float(row[1]) - get_random_wind(float(row[1]))
			R[(index3,scenario)] = 0.1*D[(index3,scenario)]
	
	for index4,row in enumerate(data4):
		index4+=1
		C[index4] = float(row[0])
		ND[index4] = int(row[1])
		V0[index4] = int(row[2])
		RU[index4] = float(row[3])
		RD[index4] = float(row[4])
		SU[index4] = float(row[5])
		SD[index4] = float(row[6])
		U0[index4] = float(row[7])
		S0[index4] = float(row[8])	
	
	#Loading the G,L values
	for j in J:
		G[j] = int(min(T, (UT[j] - U0[j])*V0[j]))
		L[j] = int(min(T, (DT[j] - S0[j])*(1-V0[j])))
	
	#Loading the K_t,j values
	for j in J:
		for t in range(1,ND[j]+1):
			if t <= tcold[j] + DT[j]:
				Kt[(t,j)] = hc[j]
			else:
				Kt[(t,j)] = cc[j]

print "loading parameters"
load_parameters()

#initial = 0 => first iteration
def iteration(iter_num):
       	global final_iter
	global Kt,J,K,OverBarP,UnderBarP,UT,DT,inistate,a,b,c,hc,cc,tcold,D
	global C,G,L,ND,R,RD,RU,S0,SD,SU,U0,V0
	global types, rho, scenarios, num_scenarios, T, best_guess, weight, solutions, problem
	global ones, pnames, overbarpnames, Kdet, Kstoch, eps 
	if iter_num == 0: 
		print "Beginning Iteration 0"
	else:
		print "Beginning Iteration ", iter_num
	quadratic_constraints = {} 

	print "Creating Variables"	
	#Block of Variables
	for scenario in scenarios:
		problem[scenario] = cplex.Cplex()
		problem[scenario].objective.set_sense(problem[scenario].objective.sense.minimize)
		problem[scenario].set_results_stream(None)
		
		quadratic_constraints[scenario] = []
		
		#Fixing initial variables
		for j in J:
			problem[scenario].variables.add(obj = [0], lb = [1.0/2.0*(UnderBarP[j]+OverBarP[j])], ub = [1.0/2.0*(UnderBarP[j]+OverBarP[j])], names = ['p'+str(j)+','+str(0)+','+str(scenario)])
			quadratic_constraints[scenario].append([['p'+str(j)+","+str(0)+","+str(scenario)],[0]])
			
			problem[scenario].variables.add(obj = [0], lb = [V0[j]], ub = [V0[j]], names = ['v'+str(j)+','+str(0)+','+str(scenario)])
			quadratic_constraints[scenario].append([['v'+str(j)+","+str(0)+","+str(scenario)],[0]])

		#The first half of the variables are taken to be deterministic
		for k in Kstoch:
			for j in J:
                        	problem[scenario].variables.add(obj = [1], lb = [0], names = ["cd"+str(j)+","+str(k)+","+str(scenario)])
				quadratic_constraints[scenario].append([["cd"+str(j)+","+str(k)+","+str(scenario)],[0]])

				problem[scenario].variables.add(obj = [1], lb = [0], names = ["cu"+str(j)+","+str(k)+','+str(scenario)])
				quadratic_constraints[scenario].append([["cu"+str(j)+","+str(k)+','+str(scenario)],[0]]) 
				
				problem[scenario].variables.add(obj = [b[j]], lb = [0], names = ['p'+str(j)+','+str(k)+','+str(scenario)])
				quadratic_constraints[scenario].append([['p'+str(j)+','+str(k)+','+str(scenario)],[2*c[j]]]) 

				problem[scenario].variables.add(obj = [0], lb = [0], names = ['OverBarp'+str(j)+','+str(k)+','+str(scenario)])
				quadratic_constraints[scenario].append([['OverBarp'+str(j)+','+str(k)+','+str(scenario)],[0]]) 

				problem[scenario].variables.add(obj = [a[j]], lb = [0], types = ['B'], names = ['v'+str(j)+','+str(k)+','+str(scenario)])
				quadratic_constraints[scenario].append([['v'+str(j)+','+str(k)+','+str(scenario)],[0]]) 

		#The second half of the variables are taken to be stochastic
		for k in Kdet:
			for j in J:
                                if iter_num > 0:  
					problem[scenario].variables.add(obj = [1 + weight[("cd",j,k,scenario,iter_num-1)] - rho[("cd",j,k)]*best_guess[("cd",j,k,iter_num-1)]], 
						lb = [0], names = ["cd"+str(j)+","+str(k)+","+str(scenario)])
					quadratic_constraints[scenario].append([["cd"+str(j)+","+str(k)+","+str(scenario)],[rho[("cd",j,k)]]])  
				else:
					problem[scenario].variables.add(obj = [1], lb = [0], names = ["cd"+str(j)+","+str(k)+","+str(scenario)])
					quadratic_constraints[scenario].append([["cd"+str(j)+","+str(k)+","+str(scenario)],[0]]) 
					rho[("cd",j,k)] = 1

                                if iter_num > 0:
					problem[scenario].variables.add(obj = [1 + weight[("cu",j,k,scenario,iter_num-1)] - rho[("cu",j,k)]*best_guess[("cu",j,k,iter_num-1)]], 
						lb = [0], names = ["cu"+str(j)+","+str(k)+','+str(scenario)])
                                        quadratic_constraints[scenario].append([["cu"+str(j)+","+str(k)+','+str(scenario)],[rho[("cu",j,k)]]]) 
                                else:
					problem[scenario].variables.add(obj = [1], lb = [0], names = ["cu"+str(j)+","+str(k)+','+str(scenario)])
                                        quadratic_constraints[scenario].append([["cu"+str(j)+","+str(k)+','+str(scenario)],[0]]) 
					rho[("cu",j,k)] = 1

                                if iter_num > 0:  
					problem[scenario].variables.add(obj = [b[j] + weight[('p',j,k,scenario,iter_num-1)] - rho[('p',j,k)]*best_guess[('p',j,k,iter_num-1)]], 
						lb = [0], names = ['p'+str(j)+','+str(k)+','+str(scenario)])
                                        quadratic_constraints[scenario].append([['p'+str(j)+','+str(k)+','+str(scenario)],[2*c[j]+rho[('p',j,k)]]]) 
                                else:
					problem[scenario].variables.add(obj = [b[j]], lb = [0], names = ['p'+str(j)+','+str(k)+','+str(scenario)])
                                        quadratic_constraints[scenario].append([['p'+str(j)+','+str(k)+','+str(scenario)],[2*c[j]]]) 
					rho[('p',j,k)] = b[j]
                                
				if iter_num > 0:  
					problem[scenario].variables.add(obj = [weight[("OverBarp",j,k,scenario,iter_num-1)] - rho[("OverBarp",j,k)]*best_guess[("OverBarp",j,k,iter_num-1)]], 
						lb = [0], names = ["OverBarp"+str(j)+','+str(k)+','+str(scenario)])
                                        quadratic_constraints[scenario].append([["OverBarp"+str(j)+','+str(k)+','+str(scenario)],[rho[("OverBarp",j,k)]]]) 
                                else:
					problem[scenario].variables.add(obj = [0], lb = [0], names = ['OverBarp'+str(j)+','+str(k)+','+str(scenario)])
                                        quadratic_constraints[scenario].append([["OverBarp"+str(j)+','+str(k)+','+str(scenario)],[0]]) 
					rho[("OverBarp",j,k)] = eps

                                if iter_num > 0:  
        				problem[scenario].variables.add(obj = [a[j] + weight[('v',j,k,scenario,iter_num-1)] - rho[('v',j,k)]*best_guess[('v',j,k,iter_num-1)]],
						lb = [0], types = ['B'], names = ['v'+str(j)+','+str(k)+','+str(scenario)])
                                        quadratic_constraints[scenario].append([['v'+str(j)+','+str(k)+','+str(scenario)],[rho[('v',j,k)]]])
				else:
					problem[scenario].variables.add(obj = [a[j]], lb = [0], types = ['B'], names = ['v'+str(j)+','+str(k)+','+str(scenario)])
                                        quadratic_constraints[scenario].append([['v'+str(j)+','+str(k)+','+str(scenario)],[0]])
					rho[('v',j,k)] = a[j]
	
	print "Adding constraints..."
	print "Block 1 constraints"
	#Block of Constraints
	for scenario in scenarios:
		for k in K:
			#(2)
			problem[scenario].linear_constraints.add(lin_expr = [cplex.SparsePair(ind = pnames[(k,scenario)], val = ones)], senses = "G", rhs = [D[(k,scenario)]])
			#(3)
			problem[scenario].linear_constraints.add(lin_expr = [cplex.SparsePair(ind = overbarpnames[(k,scenario)], val = ones)], senses = "G", rhs = [D[(k,scenario)] + R[(k,scenario)]])
			#(4) is the set of structural constraints below
			#(5) is the quadratic constraint
			#(6)-(11) can be ignored 
			for j in J: 
				#(14)
				problem[scenario].linear_constraints.add(lin_expr = [cplex.SparsePair(
					ind = ["cd"+str(j)+","+str(k)+","+str(scenario), 'v'+str(j)+','+str(k-1)+','+str(scenario), 'v'+str(j)+','+str(k)+','+str(scenario)], 
					val = [1, -C[j], C[j]])],
					senses = "G", rhs = [0])
				#(16)first
				problem[scenario].linear_constraints.add(lin_expr = [cplex.SparsePair(
					ind = ['v'+str(j)+','+str(k)+','+str(scenario), 'p'+str(j)+','+str(k)+','+str(scenario)],
					val = [UnderBarP[j],-1])], 
					senses = "L", rhs = [0])		
				#(16)second
				problem[scenario].linear_constraints.add(lin_expr = [cplex.SparsePair(
					ind = ['p'+str(j)+','+str(k)+','+str(scenario), 'OverBarp'+str(j)+','+str(k)+','+str(scenario)], 
					val = [1,-1])],
					senses = "L", rhs = [0])		
				#(17) 
				problem[scenario].linear_constraints.add(lin_expr = [cplex.SparsePair(
					ind = ['OverBarp'+str(j)+','+str(k)+','+str(scenario), 'v'+str(j)+','+str(k)+','+str(scenario)], 
					val = [1,-OverBarP[j]])],
					senses = "L", rhs = [0])		
				#(18)
				problem[scenario].linear_constraints.add(lin_expr = [cplex.SparsePair(
					ind = ['OverBarp'+str(j)+','+str(k)+','+str(scenario), 'p'+str(j)+','+str(k-1)+','+str(scenario), 
					'v'+str(j)+','+str(k-1)+','+str(scenario), 'v'+str(j)+','+str(k)+','+str(scenario)],
					val = [1, -1, -RU[j]+ SU[j], -SU[j] + OverBarP[j]])],
                	                senses = "L", rhs = [OverBarP[j]])
				#(19)
				if k < T:
					problem[scenario].linear_constraints.add(lin_expr = [cplex.SparsePair(
						ind = ['OverBarp'+str(j)+','+str(k)+','+str(scenario), 'v'+str(j)+','+str(k+1)+','+str(scenario), 
						'v'+str(j)+','+str(k)+','+str(scenario)],
						val = [1, -OverBarP[j]+SD[j], -SD[j]])], 
						senses = "L", rhs = [0])
				#(20)
				problem[scenario].linear_constraints.add(lin_expr = [cplex.SparsePair(
					ind =['p'+str(j)+','+str(k-1)+','+str(scenario), 'p'+str(j)+','+str(k)+','+str(scenario), 
					'v'+str(j)+','+str(k)+','+str(scenario), 'v'+str(j)+','+str(k-1)+','+str(scenario)], 
					val = [1, -1, -RD[j] + SD[j], -SD[j]+OverBarP[j]])],
					senses = "L", rhs = [OverBarP[j]])
				#(22)
				if k >= G[j] + 1 and k <= T - UT[j] + 1:
					entries = [1 for n in range(k+1,k+UT[j])]
					names = ['v'+str(j)+','+str(n) + ',' + str(scenario) for n in range(k+1,k+UT[j])]
					problem[scenario].linear_constraints.add(lin_expr = [cplex.SparsePair(
					ind = names + ['v'+str(j)+','+str(k)+','+str(scenario), 'v'+str(j)+','+str(k-1)+','+str(scenario)],
                	              	val = entries + [-UT[j]+1, UT[j]])], 
					senses = "G", rhs = [0])
				#(23)
				if k >= T - UT[j] + 2:
					entries = [1 for n in range(k+1,T+1)]
					names = ['v'+str(j)+','+str(n)+','+str(scenario) for n in range(k+1,T+1)]
					problem[scenario].linear_constraints.add(lin_expr = [cplex.SparsePair(
						ind = names + ['v'+str(j)+','+str(k)+','+str(scenario), 'v'+str(j)+','+str(k-1)+','+str(scenario)],
						val = entries + [-(T-k), T-k+1])], 
						senses = "G", rhs = [0])
				#(25)
				if k >= L[j] + 1 and k <= T-DT[j]+1:
					entries = [-1 for n in range(k+1,k+DT[j])]
					names = ['v'+str(j)+','+str(n)+','+str(scenario) for n in range(k+1,k+DT[j])]
					problem[scenario].linear_constraints.add(lin_expr = [cplex.SparsePair(
						ind = names + ['v' + str(j)+','+str(k-1)+','+str(scenario), 'v'+str(j)+','+str(k)+','+str(scenario)],
						val = entries + [-DT[j],DT[j]-1])], 
						senses = "G", rhs = [-DT[j]])
				#(26)
				if k >= T - DT[j] + 2:
					entries = [-1 for n in range(k+1,T+1)]
					names = ['v'+str(j)+','+str(n)+','+str(scenario) for n in range(k+1,T+1)]
					problem[scenario].linear_constraints.add(lin_expr = [cplex.SparsePair(
						ind = names + ['v' + str(j)+','+str(k-1)+','+str(scenario), 'v'+str(j)+','+str(k)+','+str(scenario)],
						val = entries + [-(T-k+1), T-k])], 
						senses = "G", rhs = [-(T-k+1)])
	print "Block 2 constraints"
	#Additional Blocks of Constraints 
	for scenario in scenarios:
		for k in K:
			for j in J:
				for t in range(1,ND[j]+1):
				#(12)	
					entries = []
					names = []
					for n in range(1,t+1):
						if k-n >= 0:
							entries.append(Kt[(t,j)])
							names.append('v'+str(j)+','+str(k-n)+','+str(scenario))
						problem[scenario].linear_constraints.add(lin_expr = [cplex.SparsePair(
							ind = names + ["cu"+str(j)+','+str(k)+','+str(scenario), 'v'+str(j)+','+str(k)+','+str(scenario)],
							val = entries + [1,-Kt[(t,j)]])],
							senses = "G", rhs = [0])
	print "Block 3 constraints"
	for scenario in scenarios:
		for j in J:
			#(21)
			entries = [1 for k in range(1,G[j]+1)]
			names = ['v'+str(j)+','+str(k)+','+str(scenario) for k in range(1,G[j]+1)]
			problem[scenario].linear_constraints.add(lin_expr = [cplex.SparsePair(ind = names, val = entries)], senses = "E", rhs = [G[j]])

			#(24)
			entries = [1 for k in range(1,L[j]+1)]
			names = ['v'+str(j)+','+str(k)+','+str(scenario) for k in range(1,L[j]+1)]
			problem[scenario].linear_constraints.add(lin_expr = [cplex.SparsePair(ind = names, val = entries)], senses = "E", rhs = [0])

	print "Adding Quadratic Constraints"	
	#Add quadratic term to objective function
	for scenario in scenarios:
		problem[scenario].objective.set_quadratic(quadratic_constraints[scenario])	
	
	#Solve the subproblem
	print "Solving the problems"
        for scenario in scenarios:
                problem[scenario].solve()

	#Saving the solutions 
	for scenario in scenarios:
		for j in J:
			for k in K:
				for type in types:
					solutions[(type,j,k,scenario,iter_num)] = problem[scenario].solution.get_values(type+str(j)+','+str(k)+','+str(scenario))
	
	print "Forming the best guess"
	#Form best guess
        for j in J:
                for k in Kdet:
                	for type in types:
				best_guess[(type,j,k,iter_num)] = 1/num_scenarios*sum(problem[scenario].solution.get_values(type+str(j)+','+str(k)+','+str(scenario)) for scenario in scenarios)

	#Updating rho values for better convergence speed        
	e = 100
	if iter_num == 0:
		vmax = {}
		vmin = {}
		for j in J:
			for k in Kdet:
				vmax[(j,k)]= max( solutions[('v',j,k,scenario,0)] for scenario in scenarios )		
				vmin[(j,k)]= min( solutions[('v',j,k,scenario,0)] for scenario in scenarios )
				rho[('v',j,k)] = e*rho[('v',j,k)]/(vmax[(j,k)] - vmin[(j,k)]+1)	
				for type in types:
					if type != 'v':
						rho[(type,j,k)] = e*rho[(type,j,k)]/max(1.0/num_scenarios*sum(abs(solutions[(type,j,k,scenario,0)] - best_guess[(type,j,k,0)]) for scenario in scenarios),1)
	
	print "Calculating weights"
        #Calculate weight
	for j in J:
        	for k in Kdet:
                	for scenario in scenarios:
				for type in types:
                                	if (iter_num == 0):
						weight[(type, j, k, scenario, 0)] = rho[(type,j,k)]*(problem[scenario].solution.get_values(type+str(j)+','+str(k)+','+str(scenario)) - 
							best_guess[(type,j,k,0)])
					else:
						weight[(type, j, k, scenario, iter_num)] = weight[(type, j, k, scenario, iter_num-1)] + rho[(type,j,k)]*(problem[scenario].solution.get_values(type+str(j)+','+str(k)+','+str(scenario)) - best_guess[("cd",j,k,iter_num)])
	
	print "Calculating convergence measure"
	#Calculate dist for determining convergence
	if iter_num > 0:
		dist = 1.0/num_scenarios*sum(sum(sum(sum(
			pow(problem[scenario].solution.get_values(type + str(j) + ',' + str(k) + ',' + str(scenario)) - best_guess[(type,j, k, iter_num)] , 2)for type in types) for j in J) for k in Kdet) for scenario in scenarios)
	else:
		dist = 10

	if iter_num == 0:
		for scenario in scenarios:
			print "iteration 0 solution for scenario ",scenario," is: ",problem[scenario].solution.get_objective_value()
	return dist

def main():
	global final_iter
	global Kt,J,K,OverBarP,UnderBarP,UT,DT,inistate,a,b,c,hc,cc,tcold,D
	global C,G,L,ND,R,RD,RU,S0,SD,SU,U0,V0
	global types, rho, scenarios, num_scenarios, T, best_guess, weight, solutions, problem
	global ones, pnames, overbarpnames, Kdet, Kstoch, eps 
	dist = []
	#Do iteration 0 and get convergence measure
	dist.append(iteration(0))
	epsilon = .1
	iter_num = 1;
	final_iter = 0

	while (dist[iter_num - 1] > epsilon and iter_num <= 300):
		#Do next iteration and get convergence measure
		dist.append(iteration(iter_num))
		iter_num +=1
		print "Iteration number", iter_num, "convergence measure", dist[iter_num-1], '\n'

	#Getting lower bound on solution
	for j in J:
		for k in Kdet:
			for type in types:
				#centering weights and fixing rho to 0
				for scenario in scenarios:
					weight[(type,j,k,scenario,iter_num-1)] = max(1,weight[(type,j,k,scenario,iter_num-1)] - 1.0/num_scenarios*sum(weight[(type,j,k,s,iter_num-1)] for s in scenarios))
				rho[(type,j,k)] = 0
	iteration(iter_num)

	Lbound = 0
        increment = [None for i in range (num_scenarios+1)]
        for j in J:
                for k in Kdet:
                        for scenario in scenarios:
                                for type in types:
                                        if type == "cd":
                                                increment[scenario] = (1-weight[(type,j,k,scenario,iter_num-1)])*solutions[(type,j,k,scenario,iter_num)]
                                        elif type == "cu":
                                                increment[scenario] += (1-weight[(type,j,k,scenario,iter_num-1)])*solutions[(type,j,k,scenario,iter_num)]
                                        elif type == "p":
                                                increment[scenario] += b[j]*solutions[(type,j,k,scenario,iter_num)] +  c[j]*pow(solutions[(type,j,k,scenario,iter_num)],2) - weight[(type,j,k,scenario,iter_num-1)]*solutions[(type,j,k,scenario,iter_num)]
                                        elif type == "v":
                                                increment[scenario] +=  (a[j] - weight[(type,j,k,scenario,iter_num-1)])*solutions[(type,j,k,scenario,iter_num)]
					elif type == "Overbarp":
						increment[scenario] += -weight[(type,j,k,scenario,iter_num-1)]*solutions[(type,j,k,scenario,iter_num)]

                    	Lbound += 1.0/num_scenarios*sum(increment[scenario] for scenario in scenarios)
                
		for k in Kstoch:
                        for scenario in scenarios:
                                for type in types:
                                        if type == "cd":
                                                increment[scenario] = solutions[(type,j,k,scenario,iter_num)]
                                        elif type == "cu":
                                                increment[scenario] += solutions[(type,j,k,scenario,iter_num)]
                                        elif type == "p":
                                                increment[scenario] += b[j]*solutions[(type,j,k,scenario,iter_num)] + c[j]*pow(solutions[(type,j,k,scenario,iter_num)],2)
                                        elif type == "v":
                                                increment[scenario] += a[j]*solutions[(type,j,k,scenario,iter_num)]
                        Lbound += 1.0/num_scenarios*sum(increment[scenario] for scenario in scenarios)

	#Getting upper bound on solution
	Ubound = 0
	for j in J:
		for k in Kdet:
			for scenario in scenarios:
				for type in types:
					if type == "cd":
						increment[scenario] = solutions[(type,j,k,scenario,iter_num)]
					elif type == "cu":
						increment[scenario] += solutions[(type,j,k,scenario,iter_num)]
					elif type == "p":
						increment[scenario] +=  b[j]*solutions[(type,j,k,scenario,iter_num)] +  c[j]*pow(solutions[(type,j,k,scenario,iter_num)],2)
					elif type == "v":
						increment[scenario] +=  a[j]*solutions[(type,j,k,scenario,iter_num)]
			Ubound += 1.0/num_scenarios*sum(increment[scenario] for scenario in scenarios)

		for k in Kstoch:
			for scenario in scenarios:
				for type in types:
					if type == "cd":
						increment[scenario] = solutions[(type,j,k,scenario,iter_num)]
					elif type == "cu":
						increment[scenario] += solutions[(type,j,k,scenario,iter_num)]
					elif type == "p":
						increment[scenario] += b[j]*solutions[(type,j,k,scenario,iter_num)] + c[j]*pow(solutions[(type,j,k,scenario,iter_num)],2)
					elif type == "v":
						increment[scenario] += a[j]*solutions[(type,j,k,scenario,iter_num)]
			Ubound += 1.0/num_scenarios*sum(increment[scenario] for scenario in scenarios)	
	
	#Report solution interval
	print "Solution Interval: [",Lbound,", ",Ubound,']'

main()
