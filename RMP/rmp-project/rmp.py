# RMPflow basic classes
# @author Anqi Li
# @date April 8, 2019

#Copyright (c) 2019, Georgia Tech Research Corporation
#Atlanta, Georgia 30332-0415
#All Rights Reserved

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

#2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

#3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



from matplotlib.pyplot import axis
import numpy as np

class RMPNode:
	"""
	A Generic RMP node
    """
	def __init__(self, name, parent, psi, J, J_dot, verbose=False):
		self.name = name

		self.parent = parent
		self.children = []

		# connect the node to its parent
		if self.parent:
			self.parent.add_child(self)

		# mapping/J/J_dot for the edge from the parent to the node
		self.psi = psi 
		self.J = J
		self.J_dot = J_dot

		# state
		self.x = None
		self.x_dot = None

		# RMP
		self.f = None # force
		self.a = None # acceleration
		self.M = None # inertion matrix

		self.child_accs = dict()
		self.child_accs[self.name] = []

		#self.child_counter = 0
		# print the name of the node when applying operations if true
		self.verbose = verbose


	def add_child(self, child):
		"""
		Add a child to the current node
	    """

		self.children.append(child)

	def remove_child(self, child):
		'''
		Remove a child from the current node
		'''

		self.children.remove(child)
		
	def pushforward(self):
		"""
		apply pushforward operation recursively
	    """

		if self.verbose:
			print('%s: pushforward' % self.name)

		if self.psi is not None and self.J is not None:
			self.x = self.psi(self.parent.x)
			self.x_dot = np.dot(self.J(self.parent.x), self.parent.x_dot)
			assert self.x.ndim == 2 and self.x_dot.ndim == 2

		[child.pushforward() for child in self.children]



	def pullback(self):
		"""
		apply pullback operation recursively
	    """
		child_fs = dict() # to store f transformed to the parent node

		[child.pullback() for child in self.children]

		if self.verbose:
			print('%s: pullback' % self.name)

		f = np.zeros_like(self.x, dtype='float64')
		M = np.zeros((max(self.x.shape), max(self.x.shape)),
			dtype='float64')

		for child in self.children:
			
			J_child = child.J(self.x)

			J_dot_child = child.J_dot(self.x, self.x_dot)
			
			assert J_child.ndim == 2 and J_dot_child.ndim == 2
			
			''' child_name = np.append(child_name, child.name)
			print(child.name)
			'''
			if child.f is not None and child.M is not None:
				#f += np.dot(J_child.T , (child.f - np.dot(np.dot(child.M, J_dot_child), self.x_dot)))
				#M += np.dot(np.dot(J_child.T, child.M), J_child)
					
				f_child = np.dot(J_child.T , (child.f - np.dot(np.dot(child.M, J_dot_child), self.x_dot)))
				M_child = np.dot(np.dot(J_child.T, child.M), J_child)	
				f += f_child
				M += M_child

				child_fs[child.name] = f_child
 		
		for child in self.children:
			# Calc acceleration for each leaf
			a = np.dot(np.linalg.pinv(M), child_fs[child.name])
			a = a.reshape(1,-1)
			if child.name not in self.child_accs:
				if self.verbose:
					print(f"Added {child.name}")
				self.child_accs[child.name] = [] 
				self.child_accs[child.name].append(a[0])
			else:
				self.child_accs[child.name].append(a[0])

		self.f = f
		self.M = M
		a = np.dot(np.linalg.pinv(M), f)
		self.child_accs[self.name].append(a.reshape(1,-1)[0])



class RMPRoot(RMPNode):
	"""
	A root node
	"""

	def __init__(self, name):
		RMPNode.__init__(self, name, None, None, None, None)

	def set_root_state(self, x, x_dot):
		"""
		set the state of the root node for pushforward
	    """

		assert x.ndim == 1 or x.ndim == 2
		assert x_dot.ndim == 1 or x_dot.ndim == 2

		if x.ndim == 1:
			x = x.reshape(-1, 1)
		if x_dot.ndim == 1:
			x_dot = x_dot.reshape(-1, 1)

		self.x = x
		self.x_dot = x_dot


	def pushforward(self):
		"""
		apply pushforward operation recursively
	    """

		if self.verbose:
			print('%s: pushforward' % self.name)

		[child.pushforward() for child in self.children]


	def resolve(self):
		"""
		compute the canonical-formed RMP
	    """

		if self.verbose:
			print('%s: resolve' % self.name)

		self.a = np.dot(np.linalg.pinv(self.M), self.f)
		#print("Acc.",self.a )
		return self.a


	def solve(self, x, x_dot):
		"""
		given the state of the root, solve for the controls
	    """

		self.set_root_state(x, x_dot)
		self.pushforward()
		self.pullback()
		return self.resolve()


class RMPLeaf(RMPNode):
	"""
	A leaf node
	"""

	def __init__(self, name, parent, parent_param, psi, J, J_dot, RMP_func):
		RMPNode.__init__(self, name, parent, psi, J, J_dot)
		self.RMP_func = RMP_func
		self.parent_param = parent_param
		self.leaf_acc = dict()
		self.name = name

	def eval_leaf(self):
		"""
		compute the natural-formed RMP given the state
	    """
		self.f, self.M = self.RMP_func(self.x, self.x_dot)


	def pullback(self):
		"""
		pullback at leaf node is just evaluating the RMP
	    """

		if self.verbose:
			print('%s: pullback' % self.name)

		self.eval_leaf()

	def add_child(self, child):
		print("CANNOT add a child to a leaf node")
		pass


	def update_params(self):
		"""
		to be implemented for updating the parameters
	    """
		pass


	def update(self):
		self.update_params()
		self.pushforward()
