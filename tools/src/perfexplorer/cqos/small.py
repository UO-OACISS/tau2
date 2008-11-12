import sys
import time
import commands

True = 1
False = 0
for classifier in ['mp']:

	f=open('/home/khuck/tau2/tools/src/perfexplorer/cqos/statsrecreate.big.' + classifier + '.csv', 'w')
	#f.write('lidvelocity, snesrtol, grashof, cflini, kspmaxit, gridsize, pc, ksprtol, ksp, default, recommended\n')
	f.write('lidvelocity, grashof, gridsize, ksp, pc, default, recommended\n')

	procs = '1'
	snes = 'ls'
	ksp = ''
	pc = ''
	cflini = '0.1'
	snesrtol = '1.000000e-08'
	kspmaxit = '200'
	ksprtol = '1.000000e-04'
	for lidvelocity in ['10', '20']:
		for grashof in ['100', '500', '1000']:
			#for gridx in ['16', '32']:
			for gridx in ['16']:
				gridy=gridx
				gridsize = `gridx` + 'x' + `gridy`
				#for pc in ['jacobi', 'bjacobi', 'none', 'sor', 'asm', 'cholesky']:
				getrec = 'java -cp /home/khuck/tau2/tools/src/perfexplorer/classifier.jar:/home/khuck/.ParaProf/weka.jar cqos.CQoSClassifier /tmp/classifier.nosplit.' + classifier + ' lidvelocity:' + lidvelocity + ' gridsize:' + gridx + 'x' + gridx + ' grashof:' + grashof
				# get KSP recommendation
				#print getrec
				(status, output) = commands.getstatusoutput(getrec) 
				#print output
				if output.startswith('fgmres'):
					ksp='fgmres'
				if output.startswith('gmres'):
					ksp='gmres'
				if output.startswith('bcgs'):
					ksp='bcgs'
				if output.startswith('tfqmr'):
					ksp='tfqmr'
				print ksp
				# get PC recommendation
				getrec = 'java -cp /home/khuck/tau2/tools/src/perfexplorer/classifier.jar:/home/khuck/.ParaProf/weka.jar cqos.CQoSClassifier /tmp/classifier.pc.' + classifier + ' lidvelocity:' + lidvelocity + ' gridsize:' + gridx + 'x' + gridx + ' grashof:' + grashof + ' ksp:' + ksp
				#print getrec
				(status, output) = commands.getstatusoutput(getrec) 
				#for pc in ['jacobi', 'bjacobi', 'none', 'sor', 'asm', 'cholesky']:
				#print output
				if output.startswith('jacobi'):
					pc='jacobi'
				if output.startswith('bjacobi'):
					pc='bjacobi'
				if output.startswith('none'):
					pc='none'
				if output.startswith('sor'):
					pc='sor'
				if output.startswith('asm'):
					pc='asm'
				if output.startswith('cholesky'):
					pc='cholesky'
				if output.startswith('ilu_0'):
					pc='ilu'
				if output.startswith('icc'):
					pc='icc'
					procs='1'
				print pc

				# make directories for results
				dirname = 'ex27'+'-'+ procs + '-' + 'x' + gridx + '-' + 'y' + gridy + '-' + 'lid' + lidvelocity + '-' + 'grh' + grashof + '-' + 'srtol' + snesrtol + '-' + 'krtol' + ksprtol + '-' + 'snes' + snes + '-' + 'pc' + pc
				print dirname
				createdir= 'mkdir /home/khuck/data/petsc/' + classifier + '/' + dirname
				#print createdir
				commands.getstatusoutput(createdir)
				commands.getstatusoutput(createdir + '/default')
				commands.getstatusoutput(createdir + '/' + ksp)
				# run with default solver
				mycommand = '$MPIEXEC -np ' + procs + ' /home/khuck/src/petsc/metadata/ex27 ' ' -snes_type ' + snes + ' -ksp_type gmres -pc_type ilu -lidvelocity ' + lidvelocity + ' -da_grid_x ' + gridx + ' -da_grid_y ' + gridx + ' -print -snes_monitor -grashof ' + grashof + ' -cfl_ini ' + cflini + ' -snes_rtol ' + snesrtol + ' -ksp_rtol ' + ksprtol + ' -preload off' + ' >& /home/khuck/data/petsc/' + classifier + '/' + dirname + '/default.log'
				# print mycommand
				start = time.time()
				(status, output) = 	commands.getstatusoutput(mycommand)
				end = time.time()
				default = end - start
				print 'DEFAULT: ', default
				commands.getstatusoutput('mv profile.* /home/khuck/data/petsc/' + classifier + '/' + dirname + '/default/.')
				# run with recommendation
				mycommand = '$MPIEXEC -np ' + procs + ' /home/khuck/src/petsc/metadata/ex27 ' ' -snes_type ' + snes + ' -ksp_type ' + ksp + ' -pc_type ' + pc + ' -lidvelocity ' + lidvelocity + ' -da_grid_x ' + gridx + ' -da_grid_y ' + gridx + ' -print -snes_monitor -grashof ' + grashof + ' -cfl_ini ' + cflini + ' -snes_rtol ' + snesrtol + ' -ksp_rtol ' + ksprtol + ' -preload off' + ' >& /home/khuck/data/petsc/' + classifier + '/' + dirname + '/' + classifier + '.log'
				start = time.time()
				(status, output) = 	commands.getstatusoutput(mycommand)
				end = time.time()
				recommended = end - start
				# print output
				print 'RECOMMENDED: ', recommended
				commands.getstatusoutput('mv profile.* /home/khuck/data/petsc/' + classifier + '/' + dirname + '/' + ksp + '/.')
				#f.write('lidvelocity, grashof, gridsize, ksp, pc, default, recommended\n')
				f.write(lidvelocity + ',' + grashof + ',' + gridsize + ',' + ksp + ',' + pc + ',' + `default` + ',' + `recommended` + '\n')
				f.flush()

f.close()

