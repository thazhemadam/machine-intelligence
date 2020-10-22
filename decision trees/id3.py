'''
Assume df is a pandas dataframe object of the dataset given
'''
import numpy as np
import pandas as pd
import random

'''
Calculate the entropy of a target attribute
'''
def entropy(target_attribute):
	elements, counts = np.unique(target_attribute,return_counts = True)
	entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
	return entropy

'''
Calculate the entropy of the entire dataset
	#input:pandas_dataframe
	#output:int/float/double/large
'''
def get_entropy_of_dataset(df):
	elements, counts = np.unique(df[df.columns[-1]], return_counts = True)
	total_entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
	return total_entropy


'''
Return entropy of the attribute provided as parameter
	#input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
	#output:int/float/double/large
'''
def get_entropy_of_attribute(df,attribute):
	vals, counts= np.unique(df[attribute],return_counts=True)	
	entropy_of_attribute = np.sum([(counts[i]/np.sum(counts))*entropy(df.where(df[attribute]==vals[i]).dropna()[df.columns[-1]]) for i in range(len(vals))])
	return abs(entropy_of_attribute)


'''
Return Information Gain of the attribute provided as parameter
	#input:int/float/double/large,int/float/double/large
	#output:int/float/double/large
'''
def get_information_gain(df,attribute):
	return (get_entropy_of_dataset(df) - get_entropy_of_attribute(df,attribute))


'''
Returns Attribute with highest info gain
	#input: pandas_dataframe
	#output: ({dict},'str')
	Return a tuple with the first element as a dictionary which has IG of all columns 
	and the second element as a string with the name of the column selected
'''
def get_selected_attribute(df):
	information_gains = {}

	for i in range(len(df.columns)-1):
		information_gains[df.columns[i]]=get_information_gain(df,df.columns[i])
	selected_column = max(information_gains, key = information_gains.get)

	return (information_gains,selected_column)
