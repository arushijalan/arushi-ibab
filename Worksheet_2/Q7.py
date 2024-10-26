import random 
def remove_all_occurrences():

    L = [random.randint(0, 10) for i in range(0,15)]
    print("The original list is: ", L)

    element = random.randint(0, 10) 
    print("The element to be removed is: ", element)

    P = [x for x in L if x != element] 
    print("The list after removal of element: ", P)  

def main():
    remove_all_occurrences()
    
if __name__ == "__main__":
    main()