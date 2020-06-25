#!/usr/bin/env python3
#Program written by Devin Shende on August 5th 2018

def firstPrimeAfter(x):
	import math
	current = x + 1
	sqrtno = math.sqrt(current)
	while True:
		#search for primes starting at x until it finds one
		#break once found a prime
		for potentialfactor in range(2,current):
			# start at 2 because 1 will always be a factor
			# go all the way up to the sqrt of current looking for a factor
			if current % potentialfactor == 0:
				# Found factor. not prime
				break # move on to next number
			elif potentialfactor >= sqrtno:
				print("The first prime number after {} is {}".format(x,current))
				return current
		current += 1

firstPrimeAfter(10000000)

