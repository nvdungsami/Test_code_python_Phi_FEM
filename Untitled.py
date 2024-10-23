#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function
import numpy as np
from dolfin import *
import sympy
import matplotlib.pyplot as plt
parameters['allow_extrapolation'] = True
parameters["form_compiler"]["representation"] = 'uflacs'
from mshr import *
from multiphenics import *
parameters["ghost_mode"] = "shared_facet" 


# Polynome of the spaces
polV = 1
polPhi = polV+2
polQuadr = 2*(polV+polPhi)


### Choice of the test case
# test case 1 : star
# test case 2 : rectangular
test_case = 2


# computation = 1 : order of convergence of L2 and H1 norm
# computation = 2 : conditionning
# computation = 3 : error / variation of theta_0
# computation = 4 : error / variation of sigma=gamma1=gamma2=gamma_div
# computation = 4 : condition number / variation of sigma=gamma1=gamma2=gamma_div
computation = 1


# size of the domain in the test case 2
R1 = 1
R2 = 2


# plot the solution
Plot = True


# Construction of phi
class phi_expr(UserExpression):
	def eval(self, value, x):
		value[0] = Phi(x[0],x[1])
	def value_shape(self):
		return (2,)


# Construction of phi
def Phi(xxx,yyy):
	if test_case ==1:
		rrr = (xxx**2+yyy**2)**(0.5)
		if xxx!=0:
			theta = np.arctan2(yyy,xxx) 
		if xxx==0 and yyy>0:
			theta = 0.5*np.pi 
		if xxx==0 and yyy<0:
			theta = -0.5*np.pi 
		if xxx==0 and yyy==0:
			theta = 0 
		res = rrr**4*(5.0 + 3.0*np.sin(7.0*(theta-theta0) + 7*np.pi/36.0))/2.0 - R**4
	if test_case==2:
		rot_x, rot_y = rot(xxx,yyy,-theta0-np.pi/8)
		res = np.max([abs(rot_x)/R1,abs(rot_y)/R2])-1
	return res


# Rotation in test case 2
def rot(x,y,theta):
	new_x = np.cos(theta)*x-np.sin(theta)*y
	new_y = np.sin(theta)*x+np.cos(theta)*y
	return new_x, new_y


# Function used to write in the outputs files
def output_latex(f,A,B):
	for i in range(len(A)):
		f.write('(')
		f.write(str(A[i]))
		f.write(',')
		f.write(str(B[i]))
		f.write(')\n')


def compute_exact_solution(mesh,polQuadr,V_exact,theta0):
	x, y = sympy.symbols('xx yy')
	if test_case == 1:
		xxx,yyy = rot(x,y,0.0)
		u_sympy = sympy.exp(y)*sympy.sin(x)
	if test_case == 2:
		xxx,yyy = rot(x,y,-theta0+np.pi/8)
		u_sympy = sympy.cos(np.pi*xxx/R1)*sympy.cos(np.pi*yyy/R2)
	f_sympy = -sympy.diff(sympy.diff(u_sympy, x),x)-sympy.diff(sympy.diff(u_sympy, y),y)+u_sympy
	f_expr = Expression(sympy.ccode(f_sympy).replace('xx', 'x[0]').replace('yy', 'x[1]'),degree=polQuadr,domain=mesh)
	u_expr = Expression(sympy.ccode(u_sympy).replace('xx', 'x[0]').replace('yy', 'x[1]'),degree=polQuadr,domain=mesh)
	phi_fine = phi_expr(element=V_exact.ufl_element())
	phi_fine = interpolate(phi_fine,V_exact)
	if test_case == 1:
		g = dot(grad(u_expr),grad(phi_fine))/(inner(grad(phi_fine),grad(phi_fine))**0.5)+u_expr*phi_fine
	if test_case == 2:
		g = Constant('0.0')
	return f_expr,g,u_expr


def compute_solution(i,sigma,gamma_div,gamma1,gamma2,theta0):
	# Construction of the mesh
	N = int(8*2**((i)))
	print("N: ",N)
	if test_case ==1:
		mesh_macro = RectangleMesh(Point(-0.5, -0.5), Point(0.5,0.5), N, N)
	if test_case ==2:
		size = float(1.1*(R1**2+R2**2)**(0.5))
		mesh_macro = RectangleMesh(Point(-size, -size), Point(size, size), N, N)
	print("Num of cells in the macro mesh: ",mesh_macro.num_cells())
	h_float = mesh_macro.hmax()
	V_phi = FunctionSpace(mesh_macro, 'CG',polPhi)
	phi = phi_expr(element=V_phi.ufl_element())
	phi = interpolate(phi,V_phi)
	domains = MeshFunction("size_t", mesh_macro, mesh_macro.topology().dim())
	domains.set_all(0)
	for ind in range(mesh_macro.num_cells()):
		mycell = Cell(mesh_macro,ind)
		v1x,v1y,v2x,v2y,v3x,v3y = mycell.get_vertex_coordinates()
		if phi(v1x,v1y)<=0.0 or phi(v2x,v2y)<=0.0 or phi(v3x,v3y)<=0.0 or near(phi(v1x,v1y),0.0) or near(phi(v2x,v2y),0.0) or near(phi(v3x,v3y),0.0):
			domains[ind] = 1
	mesh = SubMesh(mesh_macro, domains, 1)
	print("Num of cells in the mesh: ",mesh.num_cells())	

	# Facets and cells where we apply the ghost penalty
	mesh.init(1,2)
	vertices_sub = MeshFunction("bool", mesh, mesh.topology().dim()-2,False)
	facet_sub = MeshFunction("bool", mesh, mesh.topology().dim()-1,False)
	cell_sub = MeshFunction("bool", mesh, mesh.topology().dim(),False)	
	facet_ghost = MeshFunction("size_t", mesh, mesh.topology().dim()-1,0)
	cell_ghost = MeshFunction("size_t", mesh, mesh.topology().dim(),0)

	for mycell in cells(mesh):
		for myfacet in facets(mycell):
			v1, v2 = vertices(myfacet)
			if phi(v1.point().x(),v1.point().y())*phi(v2.point().x(),v2.point().y())<=0 or near(phi(v1.point().x(),v1.point().y())*phi(v2.point().x(),v2.point().y()),0.0):
				cell_ghost[mycell] = 1
				cell_sub[mycell] = True
				for myfacet2 in facets(mycell):
					facet_ghost[myfacet2] = 1
					facet_sub[myfacet2] = True
				for v in vertices(mycell):
					vertices_sub[v] = True
	for mycell in cells(mesh):
		if cell_ghost[mycell] == 0:
			for myfacet in facets(mycell):
				if facet_ghost[myfacet] == 1:
					facet_ghost[myfacet] = 2

	File2 = File("sub.rtc.xml/mesh_function_2.xml")
	File2 << cell_sub
	File1 = File("sub.rtc.xml/mesh_function_1.xml")
	File1 << facet_sub
	File0 = File("sub.rtc.xml/mesh_function_0.xml")
	File0 << vertices_sub

	# Count the number of facets and cell
	count = 0
	for mycell in cells(mesh):
		if cell_ghost[mycell] == 1:
			count += 1
	print("Number of cells in omega_h^Gamma: ",count)
	count = 0
	for myfacet in facets(mesh):
		if facet_ghost[myfacet] == 1:
			count += 1
	print("Number of facets used in the ghost penalty: ",count)

	# Construction of the functions spaces
	yp_res = MeshRestriction(mesh,"sub.rtc.xml")
	V_u = FunctionSpace(mesh, 'CG',polV)
	V_y = VectorFunctionSpace(mesh, 'CG',polV,2)	
	V_p = FunctionSpace(mesh, 'DG',polV-1)
	V = BlockFunctionSpace([V_u,V_y,V_p],restrict=[None,yp_res,yp_res])
	V_phi = FunctionSpace(mesh, 'CG',polPhi)
	V_exact = FunctionSpace(mesh, 'CG',polQuadr)

	# Construction of phi
	phi = phi_expr(element=V_phi.ufl_element())
	phi = interpolate(phi,V_phi)
	print("Construction of the spaces: ok")	

	# Computation of the source term
	f_expr,g,u_expr = compute_exact_solution(mesh,polQuadr,V_exact,theta0)
		
	# Initialize cell function for domains
	dx = Measure("dx")(domain = mesh,subdomain_data = cell_ghost)
	dS = Measure("dS")(domain = mesh,subdomain_data = facet_ghost)
	ds = Measure("ds")(domain = mesh)

	# Define trial functions
	uyp = BlockTrialFunction(V)
	u, y, p = block_split(uyp)

	# Define test functions
	vzq = BlockTestFunction(V)
	v,z,q = block_split(vzq)

	# Resolution
	n = FacetNormal(mesh)
	h = CellDiameter(mesh)

	# Construction of the bilinear form
	a_uv =  (inner(grad(u),grad(v)) +u*v)*dx + (gamma_div*u*v +gamma1*inner(grad(u),grad(v)))*dx(1) +sigma*avg(h)*dot(jump(grad(u),n),jump(grad(v),n))*dS(2) 
	a_yv = (gamma_div*div(y)*v +gamma1*inner(y,grad(v)))*dx(1)+ inner(y,n)*v*ds
	a_uz =  (gamma_div*u*div(z) +gamma1*inner(grad(u),z))*dx(1)
	a_yz =  (gamma_div*div(y)*div(z) +gamma1*inner(y,z)+h**(-2)*gamma2*inner(y,grad(phi))*inner(z,grad(phi)))*dx(1)
	a_yq = h**(-3)*gamma2*inner(y,grad(phi))*q*phi*dx(1)
	a_pz = h**(-3)*gamma2*p*phi*inner(z,grad(phi))*dx(1)
	a_pq = h**(-4)*gamma2*p*phi*q*phi*dx(1)
	a_uq = Constant('0.0')*u*q*dx(1)
	a_pv = Constant('0.0')*p*v*dx(1)

	# Construction of the linear form
	L_v =  f_expr*v*dx + gamma_div*f_expr*v*dx(1)
	L_z =  gamma_div*f_expr*(div(z))*dx(1)-h**(-2)*gamma2*g*inner(grad(phi),grad(phi))**(0.5)*inner(z,grad(phi))*dx(1)
	L_q =  -h**(-3)*gamma2*g*inner(grad(phi),grad(phi))**(0.5)*q*phi*dx(1)

	# assemble the block
	a = [[a_uv,a_yv,a_pv],[a_uz,a_yz,a_pz],[a_uq,a_yq,a_pq]]
	f =  [L_v,L_z,L_q]

	# SOLVE 
	A = block_assemble(a)
	F = block_assemble(f)
	U = BlockFunction(V)
	print("Assemble the matrices: ok")
	block_solve(A, U.block_vector(), F)
	print("Solve the problem: ok")
	sol = U[0]
	sol = interpolate(sol,V_u)
	approx_L2 = assemble((sol)**2*dx(0))**0.5
	exact_L2 = assemble((u_expr)**2*dx(0))**0.5
	approx_H1 = assemble(inner(grad(sol),grad(sol))*dx(0))**0.5
	exact_H1 = assemble(inner(grad(u_expr),grad(u_expr))*dx(0))**0.5
	err_L2 = assemble((sol-u_expr)**2*dx(0))**0.5/exact_L2
	err_H1 = assemble(inner(grad(sol-u_expr),grad(sol-u_expr))*dx(0))**0.5/exact_H1
	print('h: ',h_float)
	print('Relative L2 error: ',err_L2)
	print('Relative H1 error: ',err_H1)
	print('L2 norm of u_h: ',approx_L2)
	print('L2 norm of u: ',exact_L2)
	print('H1 norm of u_h: ',approx_H1)
	print('H1 norm of u: ',exact_H1)
	print('surface: ',assemble(Constant('1.0')*dx(0)))
	print('norm phi: ',assemble(grad(phi)**2*dx(0)))	
	if conditioning == True:
		A = np.matrix(A.array())
		cond = np.linalg.cond(A)
		print("Conditioning number: ",cond)
	else:
		cond = 0.0
	print('')

	# Plot and save
	if Plot == True:
		plot_sol = plot(sol)
		plt.savefig('myfig.png')
	return h_float,err_L2,err_H1,cond


if computation == 1:
	# Number of iterations
	init_Iter = 4
	Iter = 4

	# parameter of the scheme
	sigma = 0.01
	gamma1 = 10
	gamma2 = 10
	gamma_div = 0.01

	# Compute the conditioning number
	conditioning = False

	# Size of the domain
	R = 0.47
	theta0 = 0.9

	# Initialistion of the output
	size_mesh_vec = np.zeros(Iter)
	error_L2_vec = np.zeros(Iter)
	error_H1_vec = np.zeros(Iter)
	for i in range(init_Iter-1,Iter):
		print('##################')
		print('## Iteration ',i+1,'##')
		print('##################')
		h_float,err_L2,err_H1,cond = compute_solution(i,sigma,gamma_div,gamma1,gamma2,theta0)
		size_mesh_vec[i] = h_float
		error_L2_vec[i] = err_L2
		error_H1_vec[i] = err_H1	

	# Print the output vectors
	print('Vector h :',size_mesh_vec)
	print('Vector relative L2 error : ',error_L2_vec)
	print('Vector relative H1 error : ',error_H1_vec)

	#  Write the output file for latex
	f = open('output_test_case_{name0}_k_{name1}_l_{name2}_order.txt'.format(name0 = test_case,name1=polV,name2=polPhi),'w')
	f.write('relative L2 norm : \n')	
	output_latex(f,size_mesh_vec,error_L2_vec)
	f.write('\n')
	f.write('relative H1 norm : \n')	
	output_latex(f,size_mesh_vec,error_H1_vec)
	f.close()


if computation == 2:
	# Number of iterations
	init_Iter = 1
	Iter = 1

	# parameter of the scheme
	sigma = 0.01
	gamma_div = 10.0
	gamma1 = 10.0
	gamma2 = 10.0

	# Compute the conditioning number
	conditioning = True

	# Size of the domain
	R = 0.47
	theta0 = 0.0

	# Initialistion of the output
	size_mesh_vec = np.zeros(Iter)
	error_L2_vec = np.zeros(Iter)
	error_H1_vec = np.zeros(Iter)
	cond_vec = np.zeros(Iter)
	for i in range(init_Iter-1,Iter):
		print('##################')
		print('## Iteration ',i+1,'##')
		print('##################')
		h_float,err_L2,err_H1,cond = compute_solution(i,sigma,gamma_div,gamma1,gamma2,theta0)
		size_mesh_vec[i] = h_float
		error_L2_vec[i] = err_L2
		error_H1_vec[i] = err_H1	
		cond_vec[i] = cond

	# Print the output vectors
	print('Vector h :',size_mesh_vec)
	print('Vector relative L2 error : ',error_L2_vec)
	print('Vector relative H1 error : ',error_H1_vec)
	print("conditioning number",cond_vec)

	#  Write the output file for latex
	f = open('output_test_case_{name0}_k_{name1}_l_{name2}_cond.txt'.format(name0 = test_case,name1=polV,name2=polPhi),'w')
	f.write('relative L2 norm : \n')
	output_latex(f,size_mesh_vec,error_L2_vec)
	f.write('\n')
	f.write('relative H1 norm : \n')
	output_latex(f,size_mesh_vec,error_H1_vec)
	f.write('\n')
	f.write('conditioning number : \n')
	output_latex(f,size_mesh_vec,cond_vec)
	f.close()


if computation == 3:
	# Number of iterations
	init_Iter = 1
	Iter = 5
	if test_case == 1:
		max_theta = 0.9
		nb_theta = 90
	if test_case == 2:
		max_theta = 1.58
		nb_theta = 79

	# parameter of the scheme
	sigma = 0.01
	gamma_div = 10.0
	gamma1 = 10.0
	gamma2 = 10.0

	# Color for tikz
	color = ["blue","red","black","green","brown","yellow"]

	# Compute the conditioning number
	conditioning = False

	# Size of the domain
	R = 0.47

	# Initialistion of the output
	size_mesh_vec = np.zeros(Iter)
	thetaO_vec = np.arange(0.0,max_theta,max_theta/nb_theta)
	error_L2_vec = np.zeros((Iter,nb_theta))
	error_H1_vec = np.zeros((Iter,nb_theta))
	for j in range(nb_theta):
		theta0 = thetaO_vec[j]
		# compute exact solution
		for i in range(init_Iter-1,Iter):
			print('#####################')
			print('## Iteration ',i+1,',',j+1,'##')
			print('#####################')
			h_float,err_L2,err_H1,cond = compute_solution(i,sigma,gamma_div,gamma1,gamma2,theta0)
			size_mesh_vec[i] = h_float
			error_L2_vec[i,j] = err_L2
			error_H1_vec[i,j] = err_H1

	# Print the output vectors
	print('Vector h :',size_mesh_vec)
	print('Vector relative L2 error : ',error_L2_vec)
	print('Vector relative H1 error : ',error_H1_vec)

	#  Write the output file for latex
	f = open('output_test_case_{name0}_k_{name1}_l_{name2}_theta.txt'.format(name0 = test_case,name1=polV,name2=polPhi),'w')
	f.write('relative L2 norm : \n')
	for i in range(init_Iter-1,Iter):
		f.write("\\addplot[color=")
		f.write(color[i])
		f.write("] coordinates { \n ")
		output_latex(f,thetaO_vec,error_L2_vec[i,:])
		f.write(' }; \n')
	f.write('\n')
	f.write('relative H1 norm : \n')
	for i in range(init_Iter-1,Iter):
		f.write("\\addplot[color=")
		f.write(color[i])
		f.write("] coordinates { \n ")
		output_latex(f,thetaO_vec,error_H1_vec[i,:])
		f.write(' }; \n')
	f.close()


if computation == 4:
	# Number of iterations
	init_Iter = 4
	Iter = 4

	# Color for tikz
	color = ["blue","red","black","green","brown","yellow"]

	# Compute the conditioning number
	conditioning = False

	# Size of the domain
	R = 0.47
	theta0 = 0.0

	# Initialistion of the output
	size_mesh_vec = np.zeros(Iter)
	sigma_vec = [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e-0,1e1,1e2,1e3,1e4]
	error_L2_vec = np.zeros((Iter,len(sigma_vec)))
	error_H1_vec = np.zeros((Iter,len(sigma_vec)))
	for j in range(len(sigma_vec)):
		# compute exact solution
		for i in range(init_Iter-1,Iter):
			print('#####################')
			print('## Iteration ',i+1,',',j+1,'##')
			print('#####################')
			# parameter of the scheme
			sigma = sigma_vec[j]
			gamma_div = sigma
			gamma1 = sigma
			gamma2 = sigma
			h_float,err_L2,err_H1,cond = compute_solution(i,sigma,gamma_div,gamma1,gamma2,theta0)
			size_mesh_vec[i] = h_float
			error_L2_vec[i,j] = err_L2
			error_H1_vec[i,j] = err_H1

	# Print the output vectors
	print('Vector h :',size_mesh_vec)
	print('Vector relative L2 error : ',error_L2_vec)
	print('Vector relative H1 error : ',error_H1_vec)

	#  Write the output file for latex
	f = open('output_test_case_{name0}_k_{name1}_l_{name2}_sigma.txt'.format(name0 = test_case,name1=polV,name2=polPhi),'w')
	f.write('relative L2 norm : \n')
	for i in range(init_Iter-1,Iter):
		f.write("\\addplot[color=")
		f.write(color[i])
		f.write(",mark=*] coordinates { \n ")
		output_latex(f,sigma_vec,error_L2_vec[i,:])
		f.write(' }; \n')
	f.write('\n')
	f.write('relative H1 norm : \n')
	for i in range(init_Iter-1,Iter):
		f.write("\\addplot[color=")
		f.write(color[i])
		f.write(",mark=*] coordinates { \n ")
		output_latex(f,sigma_vec,error_H1_vec[i,:])
		f.write(' }; \n')
	f.close()


if computation == 5:
	# Number of iterations
	init_Iter = 2
	Iter = 4
	if test_case == 1:
		max_theta = 0.9
		nb_theta = 90
	if test_case == 2:
		max_theta = 1.58
		nb_theta = 79

	# parameter of the scheme
	sigma = 0.01
	gamma_div = 10.0
	gamma1 = 10.0
	gamma2 = 10.0

	# Color for tikz
	color = ["blue","red","black","green","brown","yellow"]

	# Compute the conditioning number
	conditioning = True

	# Size of the domain
	R = 0.47

	# Initialistion of the output
	size_mesh_vec = np.zeros(Iter)
	thetaO_vec = np.arange(0.0,max_theta,max_theta/nb_theta)
	error_L2_vec = np.zeros((Iter,nb_theta))
	error_H1_vec = np.zeros((Iter,nb_theta))
	cond_vec = np.zeros((Iter,nb_theta))
	for j in range(nb_theta):
		theta0 = thetaO_vec[j]
		# compute exact solution
		for i in range(init_Iter-1,Iter):
			print('#####################')
			print('## Iteration ',i+1,',',j+1,'##')
			print('#####################')
			h_float,err_L2,err_H1,cond = compute_solution(i,sigma,gamma_div,gamma1,gamma2,theta0)
			size_mesh_vec[i] = h_float
			error_L2_vec[i,j] = err_L2
			error_H1_vec[i,j] = err_H1
			cond_vec[i,j] = cond

	# Print the output vectors
	print('Vector h :',size_mesh_vec)
	print('Vector relative L2 error : ',error_L2_vec)
	print('Vector relative H1 error : ',error_H1_vec)
	print("conditioning number",cond_vec)

	#  Write the output file for latex
	f = open('output_test_case_{name0}_k_{name1}_l_{name2}_cond_theta.txt'.format(name0 = test_case,name1=polV,name2=polPhi),'w')
	f.write('conditionning: \n')
	for i in range(init_Iter-1,Iter):
		f.write("\\addplot[color=")
		f.write(color[i])
		f.write("] coordinates { \n ")
		output_latex(f,thetaO_vec,cond_vec[i,:])
		f.write(' }; \n')
	f.close()

