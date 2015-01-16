import cplex
import sys
import csv

def get_scenarios():
	scenariodata = []
	datafile = './PH1data.csv'
	with open(datafile, 'r') as file:
		reader = csv.reader(file)
		for row in reader:
			scenariodata.append(row)

	patients = {}
	for index,row in enumerate(scenariodata):
		patients[('ER', index+1, 'sat')] = row[0]
		patients[('ER', index+1, 'sun')] = row[1]
		patients[('ER', index+1, 'mon')] = row[2]
		patients[('ICU', index+1, 'sat')] = row[3]
		patients[('ICU', index+1, 'sun')] = row[4]
		patients[('ICU', index+1, 'mon')]= row[5]
	return patients

def initialize_model():
	global rooms, days, scenarios, patientspershift, wage, penaltycost, patients, best_guess_for_nurse, weight, nurseprob, rho
	rooms = ['ER', 'ICU']
	days = ['sat', 'sun', 'mon']
	scenarios = range(1,10)
	patientspershift = {('ER', 'ER'):10, ('ICU', 'ICU'):8, ('ER', 'ICU'):7, ('ICU', 'ER'):8}
	wage = 300
	penaltycost = 200
	best_guess_for_nurse = {}
	weight = {}
	nurseprob = {}
	rho = 100

	patients = get_scenarios()

#for iter_num > 0, modify iteration to include quadratic terms and increment the weights
def iteration(iter_num):
       	if iter_num > 0: quadratic_constraints = {} 
	
	for scenario in scenarios:
                nurseprob[scenario] = cplex.Cplex()
                nurseprob[scenario].objective.set_sense(nurseprob[scenario].objective.sense.minimize)
                nurseprob[scenario].set_results_stream(None)
		if iter_num > 0: quadratic_constraints[scenario] = []
		
		for day in days:
                        nurseprob[scenario].variables.add(obj = [penaltycost], lb = [0], names = ["ERbound" + '-' + day + '-' + str(scenario)])
			if iter_num > 0: quadratic_constraints[scenario].append([["ERbound" + '-' + day + '-' + str(scenario)],[0]])
                        
			nurseprob[scenario].variables.add(obj = [penaltycost], lb = [0], names = ["ICUbound" + '-' + day + '-' + str(scenario)])
			if iter_num > 0: quadratic_constraints[scenario].append([["ICUbound" + '-' + day + '-' + str(scenario)],[0]])
			
			for room in rooms:
				if iter_num > 0:
                                	nurseprob[scenario].variables.add(obj = [wage + weight[(day,room,scenario,iter_num-1)] - rho*best_guess_for_nurse[(day,room,iter_num-1)]], 
					lb = [0], types = ["I"],  names = ["nurse" + '-' + room + '-' + day + '-' + str(scenario)])
				else:
                                	nurseprob[scenario].variables.add(obj = [wage], lb = [0], types = ["I"],  names = ["nurse" + '-' + room + '-' + day + '-' + str(scenario)])
				if iter_num > 0: quadratic_constraints[scenario].append([["nurse" + '-' + room + '-' + day + '-' + str(scenario)],[rho]])
       				
				nurseprob[scenario].variables.add(obj = [0], lb = [0], names = ["nonmovingnurse" + '-' + room + '-' + day + '-' + str(scenario)])
				if iter_num > 0: quadratic_constraints[scenario].append([["nonmovingnurse"  + '-' + room + '-' + day + '-' + str(scenario)],[0.0]])

        for day in days:
                for scenario in scenarios:
                        nurseprob[scenario].linear_constraints.add(lin_expr = [cplex.SparsePair(ind = ["ERbound" + '-' + day + '-' + str(scenario), "nonmovingnurse" + '-' + "ER" + '-' + day + '-' +
                                        str(scenario), "nonmovingnurse" + '-' + "ICU" + '-' + day + '-' + str(scenario), "nurse" + '-' + "ICU" + '-' + day + '-' + str(scenario)],
                                        val = [1, patientspershift[('ER', 'ER')], -patientspershift[('ICU', 'ER')], patientspershift[('ICU', 'ER')]])], senses = "G",
                                        rhs = [float(patients[('ER',scenario,day)])])
                        nurseprob[scenario].linear_constraints.add(lin_expr = [cplex.SparsePair(ind = ["ICUbound" + '-' + day + '-' + str(scenario), "nonmovingnurse" + '-' + "ICU" + '-' + day + '-' +
                                        str(scenario), "nonmovingnurse" + '-' + "ER" + '-' + day + '-' + str(scenario), "nurse" + '-' + "ER" + '-' + day + '-' + str(scenario)],
                                        val = [1, patientspershift[('ICU', 'ICU')], -patientspershift[('ER', 'ICU')], patientspershift[('ER', 'ICU')]])], senses = "G",
                                        rhs = [float(patients[('ICU',scenario,day)])])
                        for room in rooms:
                                nurseprob[scenario].linear_constraints.add(lin_expr = [cplex.SparsePair(ind = ["nonmovingnurse" + '-' + room + '-' + day + '-' + str(scenario), "nurse" + '-' + room +
                                '-' + day + '-' + str(scenario)], val = [1,-1])], senses = "L", rhs = [0])
	
	if iter_num > 0:
		for scenario in scenarios:
			nurseprob[scenario].objective.set_quadratic(quadratic_constraints[scenario])	
	
	#Solve the subproblem
        for scenario in scenarios:
                nurseprob[scenario].solve()
        
	#Form best guess
        for day in days:
                for room in rooms:
                        best_guess_for_nurse[(day, room, iter_num)] = 1.0/9.0*sum(nurseprob[scenario].solution.get_values("nurse" + '-' + room + '-' + day + '-' + str(scenario)) 
			for scenario in scenarios)

        #Calculate weight
        for day in days:
                for room in rooms:
                        for scenario in scenarios:
                                if iter_num == 0:
					weight[(day, room, scenario, 0)] = rho*(nurseprob[scenario].solution.get_values("nurse" + '-' + room + '-' + 
					day + '-' + str(scenario)) - best_guess_for_nurse[(day, room, 0)]) 
				else:
					weight[(day, room, scenario, iter_num)] = weight[(day, room, scenario, iter_num-1)] + rho*(nurseprob[scenario].solution.get_values("nurse" + '-' + room + '-' + 
					day + '-' + str(scenario)) - best_guess_for_nurse[(day, room, iter_num)])
	
	#Calculate dist for determining convergence
	if iter_num > 0:
		dist = 1.0/9.0*sum(sum(sum(pow(nurseprob[scenario].solution.get_values("nurse" + '-' + room + '-' + day + '-' + str(scenario)) - best_guess_for_nurse[(day, room, iter_num)] , 2) 
			for room in rooms) for day in days) for scenario in scenarios)
	else:
		dist = 10

	#Print solution to watch convergence
	print "solutions"
	for room in rooms:
		for day in days:
			for scenario in scenarios:
				print "nurse" + '-' + room + '-' + day + '-' + str(scenario), nurseprob[scenario].solution.get_values("nurse" + '-' + room + '-' + day + '-' + str(scenario))
	return dist

def main():
	initialize_model()
	dist = []
	dist.append(iteration(0))
	epsilon = .1
	iter_num = 1;

	while (dist[iter_num - 1] > epsilon):
		dist.append(iteration(iter_num))
		iter_num +=1
		print "iteration number", iter_num, "convergence measure", dist[iter_num-1]

main()
