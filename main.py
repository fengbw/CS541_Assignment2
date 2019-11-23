from GRAPH import *
from QUANTIZATION import *
from LSH import *

def Modify_setting():
    return (None,None)

if __name__ == "__main__":
    choice = 0
    print("Initial step: randomly get a center (10 for recommendation) ")
    """
    """
    center=0
    range_input=int(input("Enter the range you want to search"))
    print("Search information:")
    while(choice != -1):
        print("Please choise a query method:")
        print("1: quantizaton-based method")
        print("2: locality sensitive hashing method")
        print("3: proximity-graph based method")
        print("-2: look at your setting")
        print("-3: modify setting")
        print("-1: close the program")
        choice = int(input("input:"))
        print(choice)
        if(choice == -3):
            center,range_input=Modify_setting()
        if(choice == -2):
            print (center, range_input)
        
        
    
