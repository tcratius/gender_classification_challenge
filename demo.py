from sklearn import tree

clf = tree.DecisionTreeClassifier()

## CHALLENGE - create 3 more classifiers...
#1 head_Circumference (forehead around and back) units cm
#2 wrist_Circumference 
#3 calf_Circumference

#[height, weight, shoe_size, head_cir, wrist_cir, calf_cir]
X = [[181, 80, 44, 24, 25, 41], [177, 70, 43, 22, 20, 43], [160, 60, 38, 18, 9, 34], [154, 54, 37, 19, 15, 36], [166, 65, 40, 20, 18, 45], [190, 90, 47, 29, 25, 38], [175, 64, 39, 25, 19, 42],
     [177, 70, 40, 24, 21, 48], [159, 55, 37, 17, 8, 32], [171, 75, 42, 22, 19, 35], [181, 85, 43, 30, 24, 39]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']


#CHALLENGE - ...and train them on our data
clf = clf.fit(X, Y)

prediction = clf.predict([[190, 70, 43, 25, 20, 35]])

#CHALLENGE compare their reusults and print the best one!

print prediction
