pwd_in_file = open('data/kanye/rockyou.txt')
pwd_out_file = open('data/kanye/output.txt')

pwd_in = pwd_in_file.read().split('\n')
pwd_out = pwd_out_file.read().split('\n')
matches = set(pwd_in) & set(pwd_out)

pwd_in_file.close()
pwd_out_file.close()

print('After 10000 Iterations: ')
print('Correctly Guessed Passwords: ' + str(len(matches)))
print('Total Guesses: ' + str(len(pwd_out)))
print('Accuracy: ' + str(len(matches)/len(pwd_out)))
print('Guessed against: ' + str(len(pwd_in)))
print('Correctly Guessed Passwords Listed: ')
print(matches)