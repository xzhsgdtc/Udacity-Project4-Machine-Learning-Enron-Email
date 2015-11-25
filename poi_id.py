#!/usr/bin/python


### In order to run this code, some one need make a little modification on the 
### tester.py return the recall and precision in the test_classifier class
### One can do that by adding   '  return recall, precision,f1  ' at the end of 
### the teset_classifier


import sys
import pickle
sys.path.append("../tools/")
import pprint as pp

from feature_format import featureFormat, targetFeatureSplit
from score import test_classifier, dump_classifier_and_data
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','bonus'] # You will need to use more features

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
#explore dataset print the basic infro
#Dataset size, structure, features
print "The size of the dataset is: ",  len(data_dict)
print ""    
print "An example of data: "
print data_dict[data_dict.keys()[0]]
print ""
print "Number of features is: ", len(data_dict[data_dict.keys()[0]])
print ""
print "Feature names: ", data_dict[data_dict.keys()[0]].keys()
#check Nan values
poiCounter = 0
salaryCounter = 0
emailAddressCounter = 0
nanTotalPaymentCounter = 0
nanTotalPaymentPOI = 0
nanMailToPoiCounter=0
nanTotalStockCounter = 0

for name in data_dict.keys():        
    if data_dict[name]["poi"]:            
        poiCounter +=  1
    if str(data_dict[name]['salary']) != 'NaN' :         
        salaryCounter +=  1
    if  data_dict[name]['email_address'] != None and data_dict[name]['email_address'] != 'NaN':
        emailAddressCounter +=  1         
    if  data_dict[name]['total_payments'] == None or data_dict[name]['total_payments'] == 'NaN':
        nanTotalPaymentCounter += 1       
    if data_dict[name]["poi"]:
        nanTotalPaymentPOI += 1
    if str(data_dict[name]['total_stock_value'])   == 'NaN':
        nanTotalStockCounter += 1    
    if str(data_dict[name]['from_this_person_to_poi'])   == 'NaN':
        nanMailToPoiCounter += 1
       
        
print "Number of nonTotalPaymentPOI: ",   nanTotalPaymentPOI , " percentage: " ,   float(nanTotalPaymentPOI)/poiCounter           
print "Number of NaN total payment: ",   nanTotalPaymentCounter , " percentage: " ,   float(nanTotalPaymentCounter)/len(data_dict)   
print  "Number of POI: ", poiCounter
print  "Number of non-NaN Salary: ", salaryCounter
print  "email address: ", emailAddressCounter
print  "Number of NaN total_stock_value: ", nanTotalStockCounter
print  "from_this_person_to_poi: ", nanMailToPoiCounter  



### Task 2: Remove outliers
# The three outliers has been identified in previous tasks
outlier_list = ['THE TRAVEL AGENCY IN THE PARK', 'TOTAL', 'LOCKHART EUGENE E']

#Total value is a clear outlier
#THE TRAVEL AGENCY IN THE PARK is clearly not a person...
#LOCKHART EUGENE E have NaN for all attributes, therefore we remove it

print 'we removed ',len(outlier_list),"outliers" 
for i in outlier_list:
    print i
    #pp.pprint(data_dict[i])    
    data_dict.pop(str(i),0)


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
### Here I tested several different conbinations of features,
### The ratio might give us a better feature
### From all those generated features, I only select 5 best
def cal_ratio(a,b):
    new_feature = a+'_vs_'+b
    if new_feature not in features_list:
        features_list.append(new_feature)
    if person[a] == 'NaN' or person[b]=='NaN':
        person[new_feature] = 'NaN'
    else:
        person[new_feature]= float(person[a])/float(person[b])
        
for person in data_dict.values():
    cal_ratio('bonus', 'salary')
    cal_ratio('total_stock_value','salary')
    cal_ratio('long_term_incentive', 'salary')
    cal_ratio('expenses', 'salary')
    cal_ratio('exercised_stock_options', 'total_stock_value')
    cal_ratio('deferral_payments', 'total_payments')
    cal_ratio('from_poi_to_this_person', 'to_messages')
    cal_ratio('from_this_person_to_poi', 'from_messages')
my_dataset = data_dict


data = featureFormat(data_dict, features_list, sort_keys = False)
labels, features = targetFeatureSplit(data)
selector = SelectKBest(f_classif, 5)
selector.fit(features, labels) 
KBest_feature_list = [features_list[i + 1] for i in selector.get_support(indices=True) ]
print '----------------------',KBest_feature_list


features_list = ['poi']+KBest_feature_list
data = featureFormat(data_dict, features_list, sort_keys = False)
print '----------------------'
print 'features we have:'
pp.pprint(features_list)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)





### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# For those need feature scaling 
Scaler = MinMaxScaler()
data[:,1:3]= Scaler.fit_transform(data[:,1:3])


lr = LogisticRegression()

#need scale
svc = LinearSVC()
#need scale
kn = KNeighborsClassifier()

dt =  DecisionTreeClassifier()

clf_list = [lr,svc,kn,dt]
max_score = 0


for clf in clf_list :
    recall,precision,f1,f2 = test_classifier(clf, my_dataset, features_list, folds = 1000)
    max_score = recall+precision
    print "recall = ", recall, "precision = ",precision,"f1 score = ", f1,"f2 score = ",f2, "max = ", max_score
    print "the method I use is :"
    print str(clf)
#    if recall > 0.3 and precision >0.3 and recall+precision>max_score:
#        max_score = recall+precision
#        print "recall = ", recall, "precision = ",precision, "max = ", max_score
#        print "the method I use is :"
#        print str(clf)
#        
 




# Provided to give you a starting point. Try a varity of classifiers.

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#crite = ['gini','entropy']
max_featur =['auto','sqrt','log2',None]
max_dep = range(2,10)
min_split = range(1,6)
max_score = 0

for f in max_featur:
    for d in max_dep:    
        for s in min_split:
            clf = DecisionTreeClassifier(max_features =f,max_depth = d,min_samples_split =s)
            recall,precision,f1,f2 = test_classifier(clf, my_dataset, features_list, folds = 1000)
            if recall > 0.3 and precision >0.3 and f1>max_score:
                max_score = f1
                print "recall = ", recall, "precision = ",precision, "max_f1 = ", max_score
                print "the parameters I use is :"
                print "max_features = ", f, "max_depth = ", d ,"min_samples_split = ", s
                dump_classifier_and_data(clf, my_dataset, features_list)
                
### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

