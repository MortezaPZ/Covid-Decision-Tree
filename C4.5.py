import pandas as pd
import math
from sklearn.metrics import f1_score
# 112 age
data_set = pd.read_csv('Covid Data.csv') #reading CSV
data_set = data_set.drop('DATE_DIED', axis=1) #deleting date_died column 
data_set = data_set.drop('ICU', axis=1) #deleting date_died column 
data_set = data_set.drop('PREGNANT', axis=1) #deleting date_died column 
data_set = data_set.drop('INTUBED', axis=1) #deleting date_died column 

class Node():
    def __init__(self, attribute="", CLASIFFICATION_FINAL=""):
        self.children = []
        self.isleaf = False
        self.isroot = False
        self.attribute = attribute
        self.CLASIFFICATION_FINAL = CLASIFFICATION_FINAL
        self.threshold = None


def C4_5(lis1, lis2, attribute):
    if attribute != "AGE":       
        n = 2
        if attribute == "MEDICAL_UNIT":
            n = 13
        CLASIFFICATION_FINAL = [[0 for i in range(7)] for j in range(n)]

        P_x_makhraj = 0
        for i in range(len(lis1)):
            for k in range(n):
                if lis1[i] == k+1:
                    CLASIFFICATION_FINAL[k][lis2[i]-1] += 1
                    P_x_makhraj += 1

        entropy_total = 0
        for i in range(7):
            P_x_soorat = 0 
            for k in range(n):
                P_x_soorat += CLASIFFICATION_FINAL[k][i]
            P_x = P_x_soorat/P_x_makhraj
            if P_x != 0:
                entropy_total += P_x*(math.log2(P_x))
        entropy_total = -entropy_total

        sum_entropy_Sv = 0
        for k in range(n):
            entropy = 0
            for i in range(7):
                P_x = 0
                if sum(CLASIFFICATION_FINAL[k]) != 0:
                    P_x = CLASIFFICATION_FINAL[k][i]/sum(CLASIFFICATION_FINAL[k])
                if P_x != 0:
                    entropy += P_x*(math.log2(P_x))
            sum_entropy_Sv += entropy*sum(CLASIFFICATION_FINAL[k])/P_x_makhraj

        splitINFO = 0
        for k in range(n):
            if sum(CLASIFFICATION_FINAL[k]) != 0:
                splitINFO -= sum(CLASIFFICATION_FINAL[k])/P_x_makhraj*math.log2(sum(CLASIFFICATION_FINAL[k])/P_x_makhraj)
        if splitINFO == 0:
            return (0, attribute)
        else:
            gain_ratio = (entropy_total+sum_entropy_Sv)/splitINFO
            return (gain_ratio, attribute)
    if attribute == "AGE":
        thresholds = [(2*i-1)/2 for i in range(1, 122)]
        gain = []
        for i in thresholds:
            lis_left = []
            lis_right = []
            for j in range(len(lis1)):
                if lis1[j] < i:
                    lis_left.append(lis2[j])
                else:
                    lis_right.append(lis2[j])

            entropy_total = 0
            for k in range(1, 8):
                if lis2.count(k)/len(lis2) != 0:
                    entropy_total -= (lis2.count(k)/len(lis2)*math.log2(lis2.count(k)/len(lis2)))

            if len(lis_left) != 0:
                entropy_left = 0
                for k in range(1, 8):
                    if lis_left.count(k)/len(lis_left) != 0:
                        entropy_left -= (lis_left.count(k)/len(lis_left)*math.log2(lis_left.count(k)/len(lis_left))) 

            if len(lis_right) != 0:
                entropy_right = 0
                for k in range(1, 8):
                    if lis_right.count(k)/len(lis_right) != 0:
                        entropy_right -= (lis_right.count(k)/len(lis_right)*math.log2(lis_right.count(k)/len(lis_right)))    

            gain.append((entropy_total-(len(lis_left)*entropy_left/len(lis2))-(len(lis_right)*entropy_right/len(lis2)), i))

        m = max(gain) 
        i = m[1]
        lis_right = []
        lis_left = []
        for j in range(len(lis1)):
                if lis1[j] < i:
                    lis_left.append(lis2[j])
                else:
                    lis_right.append(lis2[j])
        splitINFO = 0
        splitINFO -= (len(lis_left)/len(lis2)*math.log2(len(lis_left)/len(lis2)))
        splitINFO -= (len(lis_right)/len(lis2)*math.log2(len(lis_right)/len(lis2)))
        return (m[0]/splitINFO, attribute, i)   

class Decision_tree:
    def __init__(self):
        self.nodes = []
    def main(self, dataframe, a, node):

        target = dataframe["CLASIFFICATION_FINAL"]

        if len(a) != 0 and len(list(set(target))) != 1 and len(dataframe) != 0:  
        
            candidate = []
            for attribute in a:
                candidate.append(C4_5(list(dataframe[attribute]), list(dataframe["CLASIFFICATION_FINAL"]), attribute))

            root = max(candidate)
            if root[0] != 0:
                node.attribute = root[1]
                if root[1] == "AGE":
                    node.threshold = root[2]
                # print(root)
                if root[1] == "AGE":
                    dataframe_left = dataframe[dataframe["AGE"] < root[2]]
                    dataframe_right = dataframe[dataframe["AGE"] > root[2]]
                    a.remove("AGE")
                    node_left = Node()
                    node.children.append(node_left)
                    self.main(dataframe_left, a, node_left)
                    node_right = Node()
                    node.children.append(node_right)
                    self.main(dataframe_right, a, node_right)
                else:
                    n = 2
                    if root[1] == "MEDICAL_UNIT":
                        n = 13
                    for i in range(1, n+1):
                        if root[1] in a:
                            a.remove(root[1])
                        node_child = Node()
                        node.children.append(node_child)
                        self.main(dataframe[dataframe[root[1]] == i], a, node_child)
                    a.append(root[1])
                
            else:
                node.isleaf = True
                if len(dataframe) != 0:
                    CLASIFFICATION_FINAL = max(set(list(target)), key=list(target).count)
                    node.CLASIFFICATION_FINAL = CLASIFFICATION_FINAL
                return
        else:
            node.isleaf = True
            if len(dataframe) != 0:
                CLASIFFICATION_FINAL = max(set(list(target)), key=list(target).count)
                node.CLASIFFICATION_FINAL = CLASIFFICATION_FINAL

            # print("___________________")
            return
    def predict(self, sample, root):
        if root.isleaf:


            self.nodes.append(root.CLASIFFICATION_FINAL)
        else:
            if root.attribute == "AGE":
                if sample["AGE"] < root.threshold:
                    self.predict(sample, root.children[0])
                if sample["AGE"] > root.threshold:
                    self.predict(sample, root.children[1])
            else:
                n=2
                if root.attribute == "MEDICAL_UNIT":
                    n = 13
                for i in range(n):
                    if root.attribute:
                        if sample[root.attribute] == i+1:
                            self.predict(sample, root.children[i])


a = ['USMER', 'MEDICAL_UNIT', 'SEX', 'PATIENT_TYPE', 'PNEUMONIA', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR', 'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO', 'AGE'] 
node_root = Node()
node_root.isroot = True           
tree = Decision_tree()
tree.main(data_set, a, node_root)

sample = data_set.iloc[800000:1000000].to_dict(orient='records')
true_labels = data_set["CLASIFFICATION_FINAL"].iloc[800000:1000000].tolist()  # Extract true labels

for i in sample:
    tree.predict(i, node_root)

print(tree.nodes)  # Predicted labels
print(len(sample))  # Total samples
print(len(tree.nodes))  # Predicted samples count

# Filling missing predictions appropriately
for i in range(len(sample) - len(tree.nodes)):
    tree.nodes.append(3)  # Append default value for unpredicted cases

f1 = f1_score(true_labels, tree.nodes, average="weighted")
print(f1)
