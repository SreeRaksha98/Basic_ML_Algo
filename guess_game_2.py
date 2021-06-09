import random

n=random.randint(0,20)
chance=3
guess=0

for i in range(chance):
    while guess!=n:
          guess=input('enter any random number from 0 to 20:')
          if guess<n:
            print('The number you have entered is small')
          elif guess<0 or guess>20:
              print('Invalid number')
          else:
            print('The number you have entered is greater')

          guess=guess+1
          break
    if guess==n:
        print('''CORRECT ANSWER
            CONGRATULATIONS!!!!!''')

print('''you have completed your three trials
    TRY AGAIN!!!''')

print('Thanks for playing the game')

