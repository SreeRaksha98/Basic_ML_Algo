import random

correct_guess = random.randint(0,20)
i=0
while(i<3):
    var1 = int(input("Guess the number from 0 to 20 :"))
    i =i+1
    if var1 == correct_guess:
        print("YOU WON!!!!")
        break
    elif i<3 and var1 != correct_guess:
        print("TRY AGAIN")
    else:
        print("YOUR chance is over")
